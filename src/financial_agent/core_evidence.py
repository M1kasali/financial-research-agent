"""V45-source evidence-window component behind free-task and version-scope contracts.

Reuses released V11/V13 selection code, not historical answers or locked-answer prompts.
Retrieval remains the new scoped BM25 adapter: this is NOT the complete B35 pipeline.
"""

from dataclasses import asdict

from agent.schemas import RetrievalResult
from agent_team_b1.focused_reasoning_v11 import _focused_excerpt
from agent_team_b1.option_coverage_v13 import _coverage_windows, _header, option_coverage_evidence

from financial_agent.contracts import SearchRequest
from financial_agent.window_selection import coverage_windows, summarize_coverage


def validate_options(options):
    if not isinstance(options, dict) or len(options) > 8:
        raise ValueError("Options must be a dictionary of at most 8 candidate claims")
    for key, value in options.items():
        if not isinstance(key, str) or key not in "ABCDEFGH" or len(key) != 1:
            raise ValueError("Option keys must be A–H")
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError("Each option must contain 1–4000 characters")


def pack_core_evidence(corpus, task, hits, *, options=None, max_chars=6500,
                       max_documents=4, max_chunks=8, selection_strategy="legacy"):
    """Return bounded source excerpts and ordinal-to-versioned-citation mapping.

All hit text/metadata is rebuilt from the pinned corpus, never trusted from callers.
Window references identify full source chunks, not exact excerpt character offsets.
"""
    options = {} if options is None else options
    validate_options(options)
    if selection_strategy not in {"legacy", "coverage"}:
        raise ValueError("Unknown evidence selection strategy")
    for value, low, high in [(max_chars, 256, 28_000), (max_documents, 1, 40), (max_chunks, 1, 16)]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid evidence pack limits")
    if len(options) > max_chunks:
        raise ValueError("Each option needs at least one window slot")
    if len(hits) > 100:
        raise ValueError("At most 100 candidate hits allowed")
    allowed = frozenset(task.document_version_ids)
    if allowed - corpus.by_version.keys():
        raise ValueError("Unknown task document version")
    rows = []
    for hit in hits:
        text = corpus.read(hit.ref, allowed)
        chunk = corpus.chunks[hit.ref.chunk_id]
        rows.append(RetrievalResult(chunk.chunk_id, chunk.doc_id, chunk.domain, hit.score,
                                    "v45_source_window_adapter", task.query, text,
                                    {"page": chunk.page, "section": chunk.section}))
    coverage = None
    if selection_strategy == "coverage" and not options:
        chosen, coverage = coverage_windows(rows, question_text=task.query,
                                             max_documents=max_documents, max_chunks=max_chunks)
        while chosen:
            headers = [_header(n, row) for n, (row, _) in enumerate(chosen, 1)]
            fixed = sum(map(len, headers)) + len(chosen) + 2 * (len(chosen) - 1)
            if fixed + 48 * len(chosen) <= max_chars:
                break
            chosen.pop()
        if rows and not chosen:
            raise ValueError("max_chars cannot fit evidence headers")
        selected = [row for row, _ in chosen]
        rendered = None
    else:
        rendered, selected = option_coverage_evidence(
            rows, question_text=task.query, options=options, max_chars=max_chars,
            max_documents=max_documents, max_chunks=max_chunks,
        )
        chosen = _coverage_windows(rows, question_text=task.query, options=options,
                                   max_documents=max_documents, max_chunks=max_chunks)[:len(selected)]
    windows = []
    if selected:
        # Recover each EXACT visible body using the original selection/allowance rules.
        # Never split rendered text on headings: source data may contain fake headings.
        headers = [_header(n, row) for n, row in enumerate(selected, 1)]
        fixed = sum(map(len, headers)) + len(selected) + 2 * (len(selected) - 1)
        base, remainder = divmod(max_chars - fixed, len(selected))
        blocks = []
        for i, ((row, focus), actual, header) in enumerate(zip(chosen, selected, headers, strict=True)):
            if row is not actual:
                raise ValueError("Released evidence-window selection contract changed")
            excerpt = _focused_excerpt(row.evidence_text, focus_text=focus, max_chars=base + int(i < remainder))
            windows.append({"window_number": i + 1, "ref": asdict(corpus.refs[row.chunk_id]),
                            "excerpt_text": excerpt, "excerpt_is_full_chunk": excerpt == row.evidence_text.strip()})
            blocks.append(header + "\n" + excerpt)
        actual_rendered = "\n\n".join(blocks)
        if rendered is not None and actual_rendered != rendered:
            raise ValueError("Visible window reconstruction differs from released component")
        rendered = actual_rendered
    rendered = rendered or ""
    if len(rendered) > max_chars:
        raise ValueError("Evidence character budget exceeded")
    if coverage is not None:
        coverage.update(summarize_coverage(windows, coverage["requested_facets"], task.document_version_ids))
        coverage["windows_removed_by_character_budget"] = len(coverage["choices_before_character_budget"]) - len(windows)
    return {"status": "prepared" if selected else "no_evidence", "query": task.query,
            "options": dict(options), "document_version_ids": list(task.document_version_ids),
            "evidence_text": rendered, "windows": windows,
            "candidate_count": len(hits), "selected_window_count": len(selected),
            "rendered_chars": len(rendered), "max_chars": max_chars,
            "component": ("public_query_coverage_selection_with_released_v11_excerpt" if coverage is not None
                          else "released_b1_v13_option_coverage_with_v11_focused_excerpt"),
            "selection_strategy": selection_strategy,
            "effective_selection_strategy": "coverage" if coverage is not None else "legacy",
            "selection_note": "Option tasks retain original V13 selection" if options else None,
            "coverage_audit": coverage,
            "sufficiency": "not_assessed", "answer_generated": False, "model_calls": 0,
            "limitations": ["Lexical coverage is not semantic support or answer correctness",
                            "Full-chunk references, not exact excerpt-offset annotations",
                            "Per-option candidate selection does not guarantee each claim is supported",
                            "Not the complete B35 submission-generation pipeline"]}


def prepare_core_evidence(corpus, retriever, task, *, options=None, candidate_k=40, **limits):
    """A task query and optional claims; no qid, gold, answer seed or predetermined page."""
    if type(candidate_k) is not int or not 1 <= candidate_k <= 100:
        raise ValueError("candidate_k must be in [1,100]")
    options = {} if options is None else options
    validate_options(options)
    query = task.query + ("\n" + "\n".join(options.values()) if options else "")
    hits = retriever.search(task, SearchRequest(query, top_k=candidate_k))
    result = pack_core_evidence(corpus, task, hits, options=options, **limits)
    result["retrieval"] = {"strategy": "scoped_bm25f_lite_not_original_b1_blind_retriever",
                           "candidate_k": candidate_k, "local_index_queries": 1}
    return result

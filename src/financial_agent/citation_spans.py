"""Task-bound citation IDs for exact, contiguous, model-visible source spans.

No whitespace normalization, fuzzy matching, hidden-tail recovery or semantic approval.
"""

import hashlib
import json
import re
from copy import deepcopy

from financial_agent.contracts import EvidenceRef


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def visible_region(excerpt, source):
    """Map a unique exact body; V11 may add synthetic ellipses only at its edges."""
    candidates = []
    for left in (0, 1) if excerpt.startswith("…") else (0,):
        for right in (0, 1) if excerpt.endswith("…") else (0,):
            end = len(excerpt) - right
            body = excerpt[left:end]
            if len(body.strip()) < 2:
                continue
            start = source.find(body)
            if start >= 0:
                candidates.append((len(body), left, end, start, source.find(body, start + 1) >= 0))
    if not candidates:
        return None
    longest = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == longest]
    if len(best) != 1 or best[0][4]:
        return None  # Repeated or ambiguous location is not silently assigned to the first match.
    _, left, end, start, _ = best[0]
    return left, end, start


def segment_ranges(text, max_chars=400):
    """Prefer complete sentences, then line boundaries; preserve all internal characters."""
    if type(max_chars) is not int or max_chars < 2:
        raise ValueError("Span length must be at least two codepoints")
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        end = min(len(text), cursor + max_chars)
        if end < len(text):
            boundaries = [m.end() for m in re.finditer(r"[。！？!?；;]", text[cursor:end])]
            if boundaries:
                end = cursor + boundaries[-1]
            else:
                newline = text.rfind("\n", cursor, end)
                if newline > cursor:
                    end = newline + 1
            # A hard limit must not split a contiguous numeric token into invented smaller numbers.
            numeric = "0123456789.,+-/%"
            while end > cursor and text[end - 1] in numeric and text[end] in numeric:
                end -= 1
            if end == cursor:
                while end < len(text) and text[end] in numeric:
                    end += 1
                cursor = end  # An overlong number is visible, but receives no citation ID.
                continue
        trimmed = end
        while trimmed > cursor and text[trimmed - 1].isspace():
            trimmed -= 1
        if trimmed - cursor >= 2:
            yield cursor, trimmed
        cursor = end


def build_citation_catalog(pack, corpus, task):
    windows = pack["windows"]
    if len(windows) > 24:
        raise ValueError("Citation catalog supports at most 24 visible windows")
    numbers = [w["window_number"] for w in windows]
    if any(type(n) is not int or n < 1 for n in numbers) or len(set(numbers)) != len(numbers):
        raise ValueError("Unique positive window numbers required")
    identity = {"version": 1, "task_id": task.task_id, "session_id": task.session_id,
                "query": task.query, "scope": list(task.document_version_ids), "options": pack["options"],
                "windows": [{"number": w["window_number"], "ref": w["ref"], "excerpt": w["excerpt_text"]}
                            for w in windows]}
    catalog_id = "cat-" + _hash(identity)
    spans, unavailable = [], []
    for window in windows:
        ref = EvidenceRef(**window["ref"])
        source = corpus.read(ref, frozenset(task.document_version_ids))
        excerpt = window["excerpt_text"]
        if not isinstance(excerpt, str) or len(excerpt) > 28_000:
            raise ValueError("Invalid visible excerpt")
        region = visible_region(excerpt, source)
        if region is None:
            unavailable.append({"window_number": window["window_number"], "reason": "no_unique_exact_visible_region"})
            continue
        left, right, source_start = region
        for start, end in segment_ranges(excerpt[left:right]):
            quote = excerpt[left + start:left + end]
            if quote != source[source_start + start:source_start + end]:
                raise ValueError("Visible/source span mismatch")
            binding = {"window_number": window["window_number"], "excerpt_start": left + start,
                       "excerpt_end": left + end, "source_start": ref.start + source_start + start,
                       "source_end": ref.start + source_start + end, "quote": quote}
            binding["span_id"] = "sp-" + _hash({"catalog": catalog_id, **binding})[:24]
            spans.append(binding)
            if len(spans) > 192:
                raise ValueError("Citation catalog budget exceeded")
    return {"catalog_id": catalog_id, "offsets": "zero_based_half_open_unicode_codepoints_in_chunk_text",
            "spans": spans, "unavailable_windows": unavailable,
            "semantic_support": "not_assessed", "max_span_chars": 400}


def public_spans(catalog):
    return [{k: span[k] for k in ("span_id", "window_number", "quote")} for span in catalog["spans"]]


def resolve_span_candidate(candidate, catalog, pack, corpus, task):
    """Materialize original text for the existing strict candidate validator and reviewer."""
    if catalog != build_citation_catalog(pack, corpus, task):
        raise ValueError("Citation catalog, task, window or source changed")
    value = deepcopy(candidate)
    if not isinstance(value, dict):
        raise ValueError("Candidate must be an object")
    if value.get("decision") != "answered":
        return value  # Existing candidate validator enforces empty claims/options.
    claims = value.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= 8:
        raise ValueError("Invalid span candidate claims")
    spans = {s["span_id"]: s for s in catalog["spans"]}
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Invalid span claim")
        citations = claim.get("citations")
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise ValueError("Each claim needs 1–3 span IDs")
        resolved, seen = [], set()
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {"span_id"}:
                raise ValueError("Span citations accept only an ID, never model-written text or offsets")
            sid = citation["span_id"]
            if not isinstance(sid, str) or sid not in spans or sid in seen:
                raise ValueError("Unknown, stale or repeated span ID")
            seen.add(sid)
            span = spans[sid]
            resolved.append({"window_number": span["window_number"], "quote": span["quote"]})
        claim["citations"] = resolved
    return value

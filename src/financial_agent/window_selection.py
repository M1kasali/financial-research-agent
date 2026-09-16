"""Optional lexical coverage selection, using public query and scoped candidates only.

No answer labels, per-question IDs, company lists or predetermined source pages.
Coverage here is lexical presence, never semantic sufficiency or accounting validity.
"""

import re

from agent_team_b1.option_coverage_v13 import _lexical_score

NUMERAL = r"[0-9零〇一二三四五六七八九十百两]{1,6}"
ARTICLE = re.compile(rf"第\s*({NUMERAL})(?:\s*[-—–至~～]\s*第?\s*({NUMERAL}))?\s*条")
CLAUSE = re.compile(r"(?<![\d.])\d{1,3}\.\d{1,2}(?:\.\d{1,2})?(?![\d.])")
METRICS = ("营业收入", "净利润", "经营活动产生的现金流量净额", "经营活动现金流量净额")


def _number(raw):
    if raw.isascii() and raw.isdigit():
        return int(raw)
    digits = dict(zip("零〇一二三四五六七八九两", (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2), strict=True))
    total, current = 0, 0
    for char in raw:
        if char in digits:
            current = digits[char]
        elif char in "十百":
            total += (current or 1) * (10 if char == "十" else 100)
            current = 0
        else:
            return None
    return total + current


def clause_mentions(text):
    found = {}
    for match in ARTICLE.finditer(text):
        first, last = _number(match[1]), _number(match[2]) if match[2] else _number(match[1])
        if first is not None and last is not None and 1 <= first <= last <= 999 and last - first < 16:
            for number in range(first, last + 1):
                found.setdefault(f"article:{number}", match[0])
    for match in CLAUSE.finditer(text):
        found.setdefault("clause:" + match[0], match[0])
    return found


def requested_facets(query):
    facets = list(clause_mentions(query))[:32]
    for metric in METRICS:
        if metric in query:
            facets.append("metric:" + metric)
    if any(metric in query for metric in METRICS) and any(word in query for word in ("单位", "币种")):
        facets.append("financial_unit_declaration")
    return facets


def matched_facets(text, requested):
    mentions = clause_mentions(text)
    found = {}
    for facet in requested:
        if facet in mentions:
            found[facet] = mentions[facet]
        elif facet.startswith("metric:") and facet[7:] in text:
            found[facet] = facet[7:]
        elif facet == "financial_unit_declaration":
            match = re.search(r"(?:人民币|美元|港元|CNY|RMB)[^\n。;；]{0,20}(?:千元|万元|元)", text)
            if match:
                found[facet] = match[0]
    return found


def coverage_windows(rows, *, question_text, max_documents, max_chunks):
    requested = requested_facets(question_text)
    documents = list(dict.fromkeys(row.doc_id for row in rows))[:max_documents]
    pool, seen = [], set()
    for rank, row in enumerate(rows):
        # Only exact duplicate bodies in the SAME document; keep conflicting amounts/units.
        # Internal whitespace may separate financial table cells; NEVER concatenate it.
        signature = (row.doc_id, row.evidence_text.strip())
        if row.doc_id not in documents or signature in seen:
            continue
        seen.add(signature)
        matches = matched_facets(row.evidence_text, requested)
        pool.append({"row": row, "rank": rank, "matches": matches,
                     "score": _lexical_score(row.evidence_text, question_text)})
    selected, used, covered = [], set(), set()

    def add(item, reason):
        selected.append((item["row"], question_text + "\n" + " ".join(item["matches"].values())))
        used.add(item["row"].chunk_id)
        covered.update((item["row"].doc_id, facet) for facet in item["matches"])
        reasons.append({"chunk_id": item["row"].chunk_id, "reason": reason,
                        "matched_facets": sorted(item["matches"])})

    reasons = []
    # Reserve a candidate for each available document, subject to the same slot limit.
    for doc in documents:
        if len(selected) >= max_chunks:
            break
        candidates = [p for p in pool if p["row"].doc_id == doc]
        if candidates:
            add(max(candidates, key=lambda p: (len(p["matches"]), p["score"], -p["rank"])), "document_seed")
    while len(selected) < max_chunks:
        remaining = [p for p in pool if p["row"].chunk_id not in used]
        if not remaining:
            break
        def gain(item):
            return len({(item["row"].doc_id, f) for f in item["matches"]} - covered)
        best = max(remaining, key=lambda p: (gain(p), p["score"], -p["rank"]))
        if gain(best) == 0:
            # Preserve retrieval order for remaining slots; don't globally rerank all text.
            best = min(remaining, key=lambda p: p["rank"])
        add(best, "new_query_facet" if gain(best) else "retrieval_fill")
    return selected, {"requested_facets": requested, "candidate_documents": documents,
                      "choices_before_character_budget": reasons,
                      "exact_duplicate_bodies_removed": sum(r.doc_id in documents for r in rows) - len(pool)}


def summarize_coverage(windows, requested, authorized_versions):
    represented = {w["ref"]["version_id"] for w in windows}
    return {"represented_document_versions": sorted(represented),
            "unrepresented_document_versions": sorted(set(authorized_versions) - represented),
            "visible_query_facets": [{"window_number": w["window_number"],
                                      "facets": sorted(matched_facets(w["excerpt_text"], requested))} for w in windows],
            "coverage_is_lexical_not_semantic": True}

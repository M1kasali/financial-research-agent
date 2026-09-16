"""Fail-closed binding for a deliberately small annual-statement layout family.

Does not fill unknown legacy regex facts. Re-parses a complete titled statement,
explicit two-year columns and a single metric row, retaining exact source spans.
Only same-version, same-report general accounting declarations can supply units.
Unsupported layouts need more evidence/a new tested parser, not model guesses.
"""

import re
from decimal import Decimal

POLICY = "annual-statement-binding-v1"
METRICS = {"营业收入", "净利润", "归属于母公司所有者的净利润", "经营活动产生的现金流量净额"}
SCALES = {"元": Decimal(1), "千元": Decimal(1000), "万元": Decimal(10000), "亿元": Decimal(100000000)}
REPORT = re.compile(r"^(?P<entity>[^\n]{2,80}?有限公司)\s*(?P<year>20\d{2})\s*年年度报告(?:全文)?\s*$", re.M)
TITLE = re.compile(r"^\s*(?:\d+\s*[、.．]\s*)?(?P<title>(?:合并|母公司|公司)(?:利润表|现金流量表))\s*$", re.M)
HEADER = re.compile(r"^(?:附注[一二三四五六七八九十\d]*\s+)?(?P<current>20\d{2})\s*年(?:度)?\s+(?P<prior>20\d{2})\s*年(?:度)?\s*$", re.M)
GENERAL_UNIT = re.compile(
    r"本公司记账本位币和编制本财务报表所采用的货币均为人民币[,，]\s*"
    r"除有特别说明外[,，]\s*均以人民币(?P<unit>千元|万元|亿元|元)为单位表示[。.]"
)
ANNUAL = re.compile(r"本集团会计年度采用公历年度[,，]\s*即每年自1\s*月1\s*日起至12\s*月31\s*日止[。.]。?")
NUM = r"(?:\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|-?\d[\d,]*(?:\.\d+)?)"


def only(matches, label):
    found = list(matches)
    if len(found) != 1:
        raise ValueError(f"Statement binding requires exactly one {label}")
    return found[0]


def anchor(evidence_id, match, group=0):
    return {"evidence_id": evidence_id, "start": match.start(group), "end": match.end(group),
            "quote": match.group(group)}


def parse_statement(statement_id: str, context_ids: list[str], metric: str, expected_scope: str,
                    evidence: dict, read_evidence) -> list[dict]:
    if metric not in METRICS or expected_scope not in {"parent", "consolidated"}:
        raise ValueError("Unsupported metric or expected statement scope")
    if not context_ids or len(context_ids) > 3 or len(set(context_ids)) != len(context_ids):
        raise ValueError("One to three unique accounting declaration references required")
    text = read_evidence(statement_id)
    ref = evidence[statement_id]
    if statement_id in context_ids:
        raise ValueError("This parser requires a separate general accounting declaration")
    report = only(REPORT.finditer(text), "annual report identity")
    # The report identity must be the opening heading, not a mentioned subsidiary.
    if text[:report.start()].strip():
        raise ValueError("Annual report identity must be the opening heading")
    title = only(TITLE.finditer(text), "statement title")
    scope = "consolidated" if title.group("title").startswith("合并") else "parent"
    if scope != expected_scope:
        raise ValueError("Requested scope conflicts with statement title")
    # Refuse embedded segment/subsidiary/restricted-basis presentations. Ordinary
    # cash-flow row descriptions mentioning disposal of subsidiaries are not scopes.
    if re.search(r"单一客户|单个客户|分部信息|分地区|分行业|分产品|抵销前|抵消前|"
                 r"(?:重要|主要|非全资)子公司|[季半]度|半年度|季度|未经审计|单位另见|"
                 r"(?:以上|上述|下表|以下)(?:数据|财务信息).*?子公司", text):
        raise ValueError("Restricted, interim or ambiguous statement context")
    if re.search(r"美元|港元|欧元|USD|HKD|EUR", text):
        raise ValueError("Foreign or conflicting currency on statement page")
    header = only(HEADER.finditer(text), "two-year column header")
    preface = text[report.end():header.start()]
    if re.search(r"有限公司|分公司|分部|客户", preface):
        raise ValueError("Additional entity or restricted scope before table columns")
    years = [header.group("current"), header.group("prior")]
    if int(years[0]) != int(report.group("year")) or int(years[1]) != int(years[0]) - 1:
        raise ValueError("Report year and adjacent annual columns disagree")
    row_re = re.compile(r"^\s*(?:[一二三四五六七八九十]+[、.．]\s*)?" + re.escape(metric)
                        + r"\s+(?:(?P<note>\d{1,3})\s+)?(?P<a>" + NUM + r")\s+(?P<b>" + NUM + r")\s*$", re.M)
    row = only(row_re.finditer(text), "unambiguous metric row with two values")
    if row.group("note") and not header.group().startswith("附注"):
        raise ValueError("Extra numeric column cannot be assumed to be a note reference")
    if not report.end() <= title.start() < header.start() < row.start():
        raise ValueError("Statement identity/title/columns/row ordering invalid")
    # Refuse multiple occurrences of a metric row even if only one was parsable.
    row_starts = re.findall(r"^\s*(?:[一二三四五六七八九十]+[、.．]\s*)?" + re.escape(metric) + r"(?=\s|[（(])", text, re.M)
    if len(row_starts) != 1:
        raise ValueError("Ambiguous repeated metric rows")
    declarations = []
    for context_id in context_ids:
        context = read_evidence(context_id)
        if evidence[context_id].version_id != ref.version_id:
            raise PermissionError("Accounting context must use the same document version")
        identity = only(REPORT.finditer(context), "context report identity")
        if context[:identity.start()].strip() or identity.groupdict() != report.groupdict():
            raise ValueError("Context issuer/report year differs from statement")
        if "重要会计政策" not in context or "会计期间" not in context or "记账本位币" not in context:
            raise ValueError("General accounting policy headings required")
        unit = only(GENERAL_UNIT.finditer(context), "explicit general CNY unit declaration")
        annual = only(ANNUAL.finditer(context), "explicit annual calendar period")
        if re.search(r"美元|港元|欧元|USD|HKD|EUR", context):
            raise ValueError("Ambiguous foreign currency in accounting declaration")
        declarations.append((context_id, identity, unit, annual))
    units = {d[2].group("unit") for d in declarations}
    if len(units) != 1:
        raise ValueError("Conflicting accounting units")
    unit = units.pop()
    # A local exception cannot be overwritten by a distant general declaration.
    local_units = re.findall(r"(?:单位\s*[:：]?\s*(?:人民币)?\s*|以人民币)(千元|万元|亿元|元)", text)
    if any(value != unit for value in local_units):
        raise ValueError("Local statement unit conflicts with general declaration")
    if re.search(r"单位\s*[:：为是采按]|人民币|(?m:^\s*(?:金额)?单位)", text) and not local_units:
        raise ValueError("Unparsed local currency/unit notation requires review")
    anchors = {"entity": [anchor(statement_id, report, "entity")],
               "scope": [anchor(statement_id, title)], "columns": [anchor(statement_id, header)],
               "row": [anchor(statement_id, row)],
               "unit_currency": [anchor(eid, u) for eid, _, u, _ in declarations],
               "period_kind": [anchor(eid, a) for eid, _, _, a in declarations],
               "context_identity": [anchor(eid, identity) for eid, identity, _, _ in declarations]}
    output = []
    for group, year in zip(("a", "b"), years, strict=True):
        raw = row.group(group)
        # Bound precision for exact normalization under Decimal's standard context.
        if sum(ch.isdigit() for ch in raw) > 18:
            raise ValueError("Numeric precision exceeds supported statement policy")
        cleaned = raw.strip().replace(",", "").replace(" ", "")
        if cleaned.startswith("("):
            if cleaned[1:-1].startswith("-"):
                raise ValueError("Conflicting negative sign notation")
            cleaned = "-" + cleaned[1:-1]
        # Reject malformed commas and double negatives rather than normalize loosely.
        numeric = raw.strip().strip("()").strip()
        if re.fullmatch(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", numeric) is None:
            raise ValueError("Malformed numeric grouping")
        value = Decimal(cleaned)
        if not value.is_finite():
            raise ValueError("Non-finite statement number")
        output.append({"doc_id": ref.doc_id, "chunk_id": ref.chunk_id, "metric": metric,
                       "year": year, "raw_value": raw, "unit": unit,
                       "normalized_value": format(value * SCALES[unit], "f"),
                       "context": row.group(), "entity": report.group("entity"), "currency": "CNY",
                       "period_kind": "annual", "scope": scope, "version_id": ref.version_id,
                       "evidence_id": statement_id, "evidence_ids": [statement_id, *context_ids],
                       "extraction_mode": "statement_table_v1", "status": "context_bound_candidate",
                       "binding_policy": POLICY, "source_anchors": {**anchors, "value": [anchor(statement_id, row, group)]},
                       "binding_request": {"evidence_id": statement_id, "context_evidence_ids": context_ids,
                                           "metric": metric, "expected_scope": expected_scope},
                       "semantic_review": "not_human_approved"})
    return output

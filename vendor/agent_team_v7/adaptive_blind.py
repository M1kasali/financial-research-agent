"""B 榜盲检中的领域自适应 shortlist 与文档级覆盖保底。"""

from __future__ import annotations

from agent.index.bm25 import BM25SearchIndex
from agent.index.document_index import DocumentSearchIndex
from agent.retrieve.doc_first import retrieve_doc_first
from agent.retrieve.query import build_rule_queries
from agent.retrieve.retriever import Retriever
from agent.retrieve.targets import question_with_options
from agent.schemas import Question, RetrievalResult


EXPANDED_SHORTLIST_DOMAINS = frozenset({"financial_contracts", "insurance"})


class AdaptiveBlindRetriever(Retriever):
    """兼容现有 Solver 接口的 B 榜自适应检索器。"""

    def __init__(
        self,
        *args,
        expanded_blind_top_docs: int = 16,
        coverage_start_rank: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.expanded_blind_top_docs = expanded_blind_top_docs
        self.coverage_start_rank = coverage_start_rank

    def _candidate_doc_filter(
        self,
        question: Question,
        restrict_to_doc_ids: bool,
    ) -> set[str] | None:
        if restrict_to_doc_ids and question.doc_ids:
            return set(question.doc_ids)
        if self.doc_index and (not question.doc_ids or not restrict_to_doc_ids):
            top_k = adaptive_shortlist_size(
                question.domain,
                default_size=self.blind_top_docs,
                expanded_size=self.expanded_blind_top_docs,
            )
            doc_ids = self.doc_index.search_doc_ids(
                self._question_with_options(question),
                top_k=top_k,
                domain=question.domain,
            )
            return set(doc_ids) if doc_ids else None
        return None

    def retrieve(
        self,
        question: Question,
        restrict_to_doc_ids: bool = True,
    ) -> list[RetrievalResult]:
        is_blind = not question.doc_ids or not restrict_to_doc_ids
        if (
            is_blind
            and self.doc_index is not None
            and self.strategy == "doc_first_bm25f_expansion"
        ):
            return retrieve_adaptive_blind(
                self.index,
                self.doc_index,
                question,
                top_k=self.fused_top_k,
                default_shortlist_size=self.blind_top_docs,
                expanded_shortlist_size=self.expanded_blind_top_docs,
                coverage_start_rank=self.coverage_start_rank,
            )
        return super().retrieve(question, restrict_to_doc_ids=restrict_to_doc_ids)


def adaptive_shortlist_size(
    domain: str,
    *,
    default_size: int = 8,
    expanded_size: int = 16,
) -> int:
    """只对 A 榜盲检模拟中出现 Top-8 漏召回的领域扩展候选文档数。"""
    if default_size <= 0 or expanded_size < default_size:
        raise ValueError("shortlist 大小必须满足 0 < default_size <= expanded_size")
    return expanded_size if domain in EXPANDED_SHORTLIST_DOMAINS else default_size


def retrieve_adaptive_blind(
    index: BM25SearchIndex,
    doc_index: DocumentSearchIndex,
    question: Question,
    *,
    top_k: int = 30,
    default_shortlist_size: int = 8,
    expanded_shortlist_size: int = 16,
    coverage_start_rank: int = 0,
) -> list[RetrievalResult]:
    """先按领域确定候选文档数，再为候选文档保留至少一个块级入口。"""
    shortlist_size = adaptive_shortlist_size(
        question.domain,
        default_size=default_shortlist_size,
        expanded_size=expanded_shortlist_size,
    )
    combined_query = question_with_options(question)
    shortlist_doc_ids = doc_index.search_doc_ids(
        combined_query,
        top_k=shortlist_size,
        domain=question.domain,
    )
    keyword_bundles = [
        tuple(query.split())
        for query in build_rule_queries(question)
    ]
    global_results = retrieve_doc_first(
        index,
        keyword_bundles=keyword_bundles,
        top_docs=max(shortlist_size, top_k),
        top_k=top_k,
        filter_doc_ids=set(shortlist_doc_ids) if shortlist_doc_ids else None,
    )
    champions = retrieve_tail_document_champions(
        index,
        query=combined_query,
        shortlist_doc_ids=shortlist_doc_ids,
        tail_start_rank=coverage_start_rank,
    )
    merged = merge_tail_document_champions(
        global_results,
        champions,
        shortlist_doc_ids=shortlist_doc_ids,
        top_k=top_k,
        tail_start_rank=coverage_start_rank,
    )
    for result in merged:
        result.metadata["adaptive_shortlist_size"] = shortlist_size
        result.metadata["adaptive_shortlist_doc_ids"] = list(shortlist_doc_ids)
        result.source = "adaptive_blind_bm25f"
    return merged


def retrieve_tail_document_champions(
    index: BM25SearchIndex,
    *,
    query: str,
    shortlist_doc_ids: list[str],
    tail_start_rank: int = 8,
) -> list[RetrievalResult]:
    """对 shortlist 尾部逐文档取一个最佳块，避免全局 Top-K 完全抹去该文档。"""
    if tail_start_rank < 0:
        raise ValueError("tail_start_rank 不能为负数")
    champions: list[RetrievalResult] = []
    for rank, doc_id in enumerate(shortlist_doc_ids[tail_start_rank:], start=tail_start_rank):
        results = index.search(
            query,
            top_k=1,
            filter_doc_ids={doc_id},
            source="adaptive_blind:tail_champion",
            scoring_mode="bm25f_lite",
        )
        if not results:
            continue
        champion = results[0]
        champion.metadata["adaptive_shortlist_rank"] = rank + 1
        champions.append(champion)
    return champions


def merge_tail_document_champions(
    global_results: list[RetrievalResult],
    champions: list[RetrievalResult],
    *,
    shortlist_doc_ids: list[str],
    top_k: int,
    tail_start_rank: int = 8,
) -> list[RetrievalResult]:
    """先保留全局结果的文档覆盖，再为缺失尾部文档预留一个槽位。"""
    if top_k <= 0:
        return []
    tail_doc_ids = set(shortlist_doc_ids[tail_start_rank:])
    global_doc_ids = {item.doc_id for item in global_results}
    missing_tail_ids = tail_doc_ids - global_doc_ids

    reserved: list[RetrievalResult] = []
    reserved_doc_ids: set[str] = set()
    for champion in champions:
        if champion.doc_id not in missing_tail_ids:
            continue
        if champion.doc_id in reserved_doc_ids:
            continue
        reserved.append(champion)
        reserved_doc_ids.add(champion.doc_id)
    reserved = reserved[:top_k]

    global_budget = top_k - len(reserved)
    selected_global: list[RetrievalResult] = []
    selected_ids: set[str] = set()
    covered_docs: set[str] = set()
    for result in global_results:
        if len(selected_global) >= global_budget:
            break
        if result.doc_id in covered_docs or result.chunk_id in selected_ids:
            continue
        selected_global.append(result)
        selected_ids.add(result.chunk_id)
        covered_docs.add(result.doc_id)
    for result in global_results:
        if len(selected_global) >= global_budget:
            break
        if result.chunk_id in selected_ids:
            continue
        selected_global.append(result)
        selected_ids.add(result.chunk_id)

    return [*selected_global, *reserved]

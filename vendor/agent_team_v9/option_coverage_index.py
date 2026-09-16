"""只在逐选项检索调用中保留每个候选文档的最佳块。"""

from __future__ import annotations

from agent.index.bm25 import BM25SearchIndex
from agent.schemas import RetrievalResult


OPTION_SOURCE_PREFIXES = ("option_", "multi_logicrag_retry_")


class OptionCoverageIndex:
    """透明包装 BM25 索引，为现有多选 Solver 注入文档覆盖候选。"""

    def __init__(
        self,
        delegate: BM25SearchIndex,
        *,
        pool_multiplier: int = 8,
    ) -> None:
        if pool_multiplier <= 0:
            raise ValueError("pool_multiplier 必须为正数")
        self.delegate = delegate
        self.pool_multiplier = pool_multiplier

    @property
    def default_search_mode(self) -> str:
        return self.delegate.default_search_mode

    @default_search_mode.setter
    def default_search_mode(self, value: str) -> None:
        self.delegate.default_search_mode = value

    def __getattr__(self, name: str):
        return getattr(self.delegate, name)

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_doc_ids: set[str] | None = None,
        filter_chunk_types: set[str] | None = None,
        source: str = "bm25",
        scoring_mode: str | None = None,
    ) -> list[RetrievalResult]:
        if not self._requires_option_coverage(source, filter_doc_ids, top_k):
            return self.delegate.search(
                query,
                top_k=top_k,
                filter_doc_ids=filter_doc_ids,
                filter_chunk_types=filter_chunk_types,
                source=source,
                scoring_mode=scoring_mode,
            )

        requested_docs = set(filter_doc_ids or ())
        pool_top_k = max(top_k, len(requested_docs) * self.pool_multiplier)
        expanded = self.delegate.search(
            query,
            top_k=pool_top_k,
            filter_doc_ids=requested_docs,
            filter_chunk_types=filter_chunk_types,
            source=source,
            scoring_mode=scoring_mode,
        )
        champion_by_doc: dict[str, RetrievalResult] = {}
        for result in expanded:
            champion_by_doc.setdefault(result.doc_id, result)

        for doc_id in sorted(requested_docs - set(champion_by_doc)):
            fallback = self.delegate.search(
                query,
                top_k=1,
                filter_doc_ids={doc_id},
                filter_chunk_types=filter_chunk_types,
                source=source,
                scoring_mode=scoring_mode,
            )
            if not fallback:
                continue
            item = fallback[0]
            item.metadata["option_coverage_raw_score"] = item.score
            item.score = 0.0
            champion_by_doc[doc_id] = item

        champions = sorted(
            champion_by_doc.values(),
            key=lambda item: (-float(item.score), item.doc_id, item.chunk_id),
        )
        for item in champions:
            item.metadata["option_coverage_candidate"] = True
        selected = champions[:top_k]
        selected_ids = {item.chunk_id for item in selected}
        for result in expanded:
            if len(selected) >= top_k:
                break
            if result.chunk_id in selected_ids:
                continue
            selected.append(result)
            selected_ids.add(result.chunk_id)
        return selected

    @staticmethod
    def _requires_option_coverage(
        source: str,
        filter_doc_ids: set[str] | None,
        top_k: int,
    ) -> bool:
        return bool(
            filter_doc_ids
            and len(filter_doc_ids) <= top_k
            and source.startswith(OPTION_SOURCE_PREFIXES)
        )

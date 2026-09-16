"""Scope-enforced adapter around the existing lexical retrieval implementation."""

from agent.index.bm25 import BM25SearchIndex

from financial_agent.contracts import EvidenceHit, EvidenceRef, SearchRequest, TaskRequest
from financial_agent.corpus import Corpus


class ScopedRetriever:
    def __init__(self, corpus: Corpus, *, tokenizer_mode: str = "mixed"):
        if tokenizer_mode not in {"mixed", "char", "word"}:
            raise ValueError("Unsupported tokenizer mode")
        self.corpus = corpus
        self.tokenizer_mode = tokenizer_mode
        self.index = BM25SearchIndex.build(list(corpus.chunks.values()), tokenizer_mode=tokenizer_mode)

    def search(self, task: TaskRequest, request: SearchRequest) -> list[EvidenceHit]:
        allowed = frozenset(task.document_version_ids)
        if allowed - self.corpus.by_version.keys():
            raise ValueError("Task references unknown document versions")
        requested = (
            allowed if request.document_version_ids is None else frozenset(request.document_version_ids)
        )
        if requested - allowed:
            raise PermissionError("Requested documents exceed task scope")
        # Legacy BM25 treats empty filters as unrestricted. Never pass an empty set.
        if not requested:
            return []
        doc_ids = {self.corpus.by_version[v].doc_id for v in requested}
        rows = self.index.search(
            request.query,
            top_k=request.top_k,
            filter_doc_ids=doc_ids,
            source="scoped_bm25",
            scoring_mode="bm25f_lite",
        )
        results = []
        for row in rows:
            ref = self.corpus.refs.get(row.chunk_id)
            if ref is None or ref.version_id not in requested:
                raise PermissionError("Retriever returned out-of-scope evidence")
            text = self.corpus.read(ref, requested)
            results.append(EvidenceHit(ref, text, row.score, self.corpus.chunks[row.chunk_id].section))
        return results

    def search_balanced(self, task: TaskRequest, request: SearchRequest) -> list[EvidenceHit]:
        """Round-robin document candidates; no cross-document score comparability assumed.

        One bounded logical tool, at most four local index queries. Does not guarantee
        useful evidence in every document or that returned hits answer the question.
        """
        versions = task.document_version_ids if request.document_version_ids is None else request.document_version_ids
        if len(set(versions)) != len(versions):
            raise ValueError("Duplicate document scope")
        if not set(versions) <= set(task.document_version_ids):
            raise PermissionError("Requested documents exceed task scope")
        if not versions:
            return []
        if not 1 <= len(versions) <= 4 or not len(versions) <= request.top_k <= 8:
            raise ValueError("Balanced search requires 1-4 documents and document_count <= top_k <= 8")
        pools = [self.search(task, SearchRequest(request.query, (v,), request.top_k)) for v in versions]
        return self._interleave(pools, request.top_k)

    @staticmethod
    def _interleave(pools: list[list[EvidenceHit]], top_k: int) -> list[EvidenceHit]:
        output, seen = [], set()
        # Exact-text duplicates are redundant only WITHIN the same document version.
        queues = [iter(pool) for pool in pools]
        while len(output) < top_k:
            progressed = False
            for queue in queues:
                for hit in queue:
                    key = (hit.ref.version_id, hit.ref.text_sha256)
                    if key not in seen:
                        seen.add(key)
                        output.append(hit)
                        progressed = True
                        break
                if len(output) == top_k:
                    return output
            if not progressed:
                break
        return output

    def context_candidates(
        self, task: TaskRequest, seed: EvidenceRef, query: str, *, radius: int = 1, top_k: int = 6
    ) -> list[EvidenceHit]:
        """Read same-version nearby pages plus query-selected, possibly distant pages.

        Context is evidence to inspect, NEVER automatically inherited accounting facts.
        No caller-supplied document ID or page can switch away from the seed's version.
        """
        self.corpus.read(seed, frozenset(task.document_version_ids))
        if type(radius) is not int or not 0 <= radius <= 2:
            raise ValueError("Context radius must be in [0,2]")
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError("Context top_k must be in [1,8]")
        ranked = self.search(task, SearchRequest(query, (seed.version_id,), top_k + 1))
        ranked = [hit for hit in ranked if hit.ref.text_sha256 != seed.text_sha256]
        nearby = []
        if seed.physical_page is not None:
            refs = [ref for ref in self.corpus.refs.values()
                    if ref.version_id == seed.version_id and ref.physical_page is not None
                    and abs(ref.physical_page - seed.physical_page) <= radius
                    and ref.text_sha256 != seed.text_sha256]
            refs.sort(key=lambda ref: (abs(ref.physical_page - seed.physical_page),
                                       ref.physical_page, ref.chunk_id))
            for ref in refs[:top_k]:
                nearby.append(EvidenceHit(ref, self.corpus.read(ref, frozenset({seed.version_id})),
                                          0.0, self.corpus.chunks[ref.chunk_id].section))
        # Give targeted query and neighboring-page candidates room, rather than
        # letting a large number of neighboring table fragments consume every slot.
        return self._interleave([ranked, nearby], top_k)

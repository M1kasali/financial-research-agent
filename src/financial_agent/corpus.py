"""Versioned local JSONL corpus. No model requests or pickle deserialization."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.schemas import Chunk

from financial_agent.contracts import EvidenceRef


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentVersion:
    doc_id: str
    version_id: str
    domain: str
    title: str
    chunk_count: int


class Corpus:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("Corpus is empty")
        self.chunks: dict[str, Chunk] = {}
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if any(
                not isinstance(value, str) or not value.strip()
                for value in (chunk.chunk_id, chunk.doc_id, chunk.text, chunk.domain)
            ):
                raise ValueError("Each chunk needs non-empty IDs and text")
            if chunk.page is not None and (type(chunk.page) is not int or chunk.page < 1):
                raise ValueError("physical_page must be a positive 1-based integer or None")
            if chunk.chunk_id in self.chunks:
                raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}")
            self.chunks[chunk.chunk_id] = chunk
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        self.documents: dict[str, DocumentVersion] = {}
        self.by_version: dict[str, DocumentVersion] = {}
        for doc_id, members in sorted(grouped.items()):
            if len({c.domain for c in members}) != 1:
                raise ValueError(f"Conflicting domains for document: {doc_id}")
            digest = hashlib.sha256()
            for chunk in sorted(members, key=lambda c: c.chunk_id):
                encoded = json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                digest.update(encoded.encode("utf-8") + b"\n")
            record = DocumentVersion(
                doc_id,
                f"{doc_id}@{digest.hexdigest()}",
                members[0].domain,
                str(members[0].metadata.get("title") or doc_id),
                len(members),
            )
            self.documents[doc_id] = record
            self.by_version[record.version_id] = record
        # Pin original evidence bindings before legacy index construction mutates metadata.
        self.refs = {cid: self._make_ref(chunk) for cid, chunk in self.chunks.items()}
        self.source_metadata = {cid: deepcopy(chunk.metadata) for cid, chunk in self.chunks.items()}

    @classmethod
    def from_jsonl(cls, path: Path) -> "Corpus":
        chunks = []
        with path.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, start=1):
                if line.strip():
                    try:
                        chunks.append(Chunk.from_dict(json.loads(line)))
                    except (ValueError, TypeError, KeyError) as exc:
                        raise ValueError(f"Invalid chunk record on line {number}: {exc}") from exc
        return cls(chunks)

    def _make_ref(self, chunk: Chunk) -> EvidenceRef:
        version_id = self.documents[chunk.doc_id].version_id
        text_hash = sha256_text(chunk.text)
        evidence_id = sha256_text(json.dumps([version_id, chunk.chunk_id, 0, len(chunk.text), text_hash]))
        return EvidenceRef(
            evidence_id, chunk.doc_id, version_id, chunk.chunk_id, chunk.page, 0, len(chunk.text), text_hash
        )

    def read(self, ref: EvidenceRef, allowed_versions: frozenset[str]) -> str:
        if ref.version_id not in allowed_versions:
            raise PermissionError("Evidence is outside task document scope")
        expected = self.refs.get(ref.chunk_id)
        if expected != ref:
            raise ValueError("Unknown or modified evidence reference")
        text = self.chunks[ref.chunk_id].text
        if sha256_text(text) != ref.text_sha256:
            raise ValueError("Source text changed after version binding")
        return text[ref.start : ref.end]

"""Vector database retrieval using ChromaDB and BAAI/bge-m3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, List, Sequence, TypeVar, Optional

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

import chromadb
from chromadb.utils import embedding_functions
import logging
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

CrossEncoder = None

T = TypeVar("T")


@dataclass(frozen=True)
class SearchHit(Generic[T]):
    item: T
    score: float
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None
        if CrossEncoder is not None:
            # Lazy loading to save memory
            self._load_model()

    def _load_model(self):
        if self.model is None and CrossEncoder is not None:
            self.model = CrossEncoder(self.model_name, max_length=512)

    def rerank(self, query: str, hits: List[SearchHit[T]], text_fn: Callable[[T], str], top_k: int = 3) -> List[SearchHit[T]]:
        if not hits:
            return []
        
        # If model cannot be loaded, fallback to original ranking
        if self.model is None:
            return hits[:top_k]

        pairs = [[query, text_fn(hit.item)] for hit in hits]
        scores = self.model.predict(pairs)
        
        reranked_hits = []
        for i, hit in enumerate(hits):
            reranked_hits.append((scores[i], hit))
            
        reranked_hits.sort(key=lambda x: x[0], reverse=True)
        
        final_hits = []
        for rank, (score, original_hit) in enumerate(reranked_hits[:top_k]):
            # Use a normalized sigmoid score or just the raw cross-encoder score
            # bge-reranker outputs logits, which can be negative or positive.
            final_hits.append(SearchHit(item=original_hit.item, score=float(score), rank=rank + 1))
            
        return final_hits


class VectorDBIndex(Generic[T]):
    def __init__(
        self,
        collection_name: str,
        items: Sequence[T],
        text_fn: Callable[[T], str],
        id_fn: Callable[[T], str],
        db_path: str,
    ) -> None:
        self.items = list(items)
        self.items_by_id = {id_fn(item): item for item in self.items}
        self.text_fn = text_fn
        self.id_fn = id_fn
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        # Use a lightweight embedding model (~80MB, fast, low RAM)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Append suffix to avoid dimension mismatch with the old BAAI/bge-m3 collection
        collection_name = collection_name + "_bgesmall"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        if self.collection.count() == 0 and self.items:
            print(f"Indexing {len(self.items)} items into collection '{collection_name}'...")
            batch_size = 100
            for i in range(0, len(self.items), batch_size):
                batch_items = self.items[i:i + batch_size]
                self.collection.upsert(
                    documents=[text_fn(item) for item in batch_items],
                    ids=[id_fn(item) for item in batch_items]
                )
            print(f"Indexing complete for '{collection_name}'.")

    def search(self, query: str, k: int = 50, reranker: Optional[CrossEncoderReranker] = None, rerank_top_k: int = 3) -> List[SearchHit[T]]:
        if self.collection.count() == 0:
            return []
            
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count())
        )
        
        hits = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else []
            
            for rank, (item_id, distance) in enumerate(zip(ids, distances)):
                item = self.items_by_id.get(item_id)
                if item:
                    score = 1.0 - distance
                    hits.append(SearchHit(item=item, score=score, rank=rank + 1))
                    
        if reranker is not None:
            hits = reranker.rerank(query, hits, self.text_fn, top_k=rerank_top_k)
        else:
            hits = hits[:rerank_top_k] if rerank_top_k else hits
            
        return hits

    @classmethod
    def from_items(
        cls, 
        collection_name: str, 
        items: Sequence[T], 
        text_fn: Callable[[T], str], 
        id_fn: Callable[[T], str],
        db_path: str
    ) -> "VectorDBIndex[T]":
        return cls(collection_name, items, text_fn, id_fn, db_path)


def render_hits(hits: Iterable[SearchHit], text_fn: Callable[[object], str], max_chars: int = 900) -> str:
    parts: List[str] = []
    for hit in hits:
        text = text_fn(hit.item).replace("\n", " ")
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        parts.append(f"[rank={hit.rank} score={hit.score:.3f}] {text}")
    return "\n".join(parts)

"""Hybrid Database using ChromaDB and NetworkX."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import networkx as nx

from exact_pipeline.knowledge.retrieval import SearchHit, VectorDBIndex


@dataclass(frozen=True)
class HybridSearchHit:
    item: Any
    score: float
    rank: int
    vector_score: float
    pagerank_score: float
    is_fast_path: bool = False


class HybridDB:
    def __init__(
        self,
        collection_name: str,
        items: Sequence[Any],
        text_fn: Callable[[Any], str],
        id_fn: Callable[[Any], str],
        db_path: str,
        graph_path: str,
        alpha: float = 0.15,
    ) -> None:
        self.vector_db = VectorDBIndex.from_items(
            collection_name=collection_name,
            items=items,
            text_fn=text_fn,
            id_fn=id_fn,
            db_path=db_path,
        )
        self.graph_path = graph_path
        self.alpha = alpha
        self.id_fn = id_fn
        self.text_fn = text_fn

        if os.path.exists(self.graph_path):
            self.graph = nx.read_graphml(self.graph_path)
            # Re-populate vector_db.items_by_id with dynamic items from the graph
            for node_id, data in self.graph.nodes(data=True):
                if node_id not in self.vector_db.items_by_id:
                    # Create a dummy FormulaCard or Logic card equivalent using the stored text
                    from exact_pipeline.knowledge.knowledge import FormulaCard
                    if "text" in data:
                        text = data["text"]
                        # We just need an object with a .render() method or just a string
                        # Let's create a generic item that works with the existing codebase
                        if "physics" in collection_name:
                            item = FormulaCard(node_id, "dynamic", text, "", ("dynamic",))
                        else:
                            item = text
                        self.vector_db.items.append(item)
                        self.vector_db.items_by_id[node_id] = item
        else:
            self.graph = nx.DiGraph()
            # Initialize with base items
            for item in items:
                item_id = id_fn(item)
                self.graph.add_node(item_id, text=text_fn(item), is_dynamic=False)
            self.save_graph()

        self.pagerank: Dict[str, float] = {}
        self._update_pagerank()

    def _update_pagerank(self) -> None:
        if len(self.graph) > 0:
            try:
                self.pagerank = nx.pagerank(self.graph, alpha=0.85)
            except nx.NetworkXError:
                self.pagerank = {node: 1.0 / len(self.graph) for node in self.graph.nodes()}
        else:
            self.pagerank = {}

    def save_graph(self) -> None:
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        nx.write_graphml(self.graph, self.graph_path)

    def add_rule(self, rule_id: str, item: Any, dependencies: Optional[List[str]] = None, auto_save: bool = True) -> None:
        text = self.text_fn(item)
        self.graph.add_node(rule_id, text=text, verified_by_executor=True, is_dynamic=True)
        if dependencies:
            for dep in dependencies:
                if self.graph.has_node(dep):
                    self.graph.add_edge(dep, rule_id)
        
        if auto_save:
            self.save_graph()
            self._update_pagerank()
        
        self.vector_db.collection.upsert(documents=[text], ids=[rule_id])
        self.vector_db.items.append(item)
        self.vector_db.items_by_id[rule_id] = item

    def search(self, query: str, k: int = 5) -> List[HybridSearchHit]:
        # Search Vector DB
        vector_hits = self.vector_db.search(query, k=k)

        hybrid_hits = []
        for v_hit in vector_hits:
            item_id = self.id_fn(v_hit.item)
            pr_score = self.pagerank.get(item_id, 0.0)

            # Master Plan formula: score = alpha * cosine_sim + (1-alpha) * pagerank
            final_score = self.alpha * v_hit.score + (1.0 - self.alpha) * pr_score

            hybrid_hits.append(
                HybridSearchHit(
                    item=v_hit.item,
                    score=final_score,
                    rank=0,
                    vector_score=v_hit.score,
                    pagerank_score=pr_score,
                )
            )

        # Sort by hybrid score descending
        hybrid_hits.sort(key=lambda x: x.score, reverse=True)
        for i, hit in enumerate(hybrid_hits):
            hybrid_hits[i] = HybridSearchHit(
                item=hit.item,
                score=hit.score,
                rank=i + 1,
                vector_score=hit.vector_score,
                pagerank_score=hit.pagerank_score,
            )

        return hybrid_hits[:k]

    def extract_reasoning_subgraph(self, start_nodes: List[str], max_depth: int = 2) -> nx.DiGraph:
        """Extract a subgraph focusing on reasoning paths around the provided nodes."""
        if not self.graph or not start_nodes:
            return nx.DiGraph()
            
        subgraph_nodes = set(start_nodes)
        
        # Add neighbors up to max_depth
        for node in start_nodes:
            if not self.graph.has_node(node):
                continue
                
            # Forward paths (dependencies)
            try:
                edges = nx.bfs_edges(self.graph, source=node, depth_limit=max_depth)
                for u, v in edges:
                    subgraph_nodes.add(u)
                    subgraph_nodes.add(v)
            except nx.NetworkXError:
                pass
                
            # Backward paths (what depends on this)
            if self.graph.is_directed():
                try:
                    edges = nx.bfs_edges(self.graph.reverse(), source=node, depth_limit=max_depth)
                    for u, v in edges:
                        subgraph_nodes.add(u)
                        subgraph_nodes.add(v)
                except nx.NetworkXError:
                    pass
                    
        return self.graph.subgraph(subgraph_nodes)

    def search_with_subgraph(self, query: str, k: int = 5, max_depth: int = 2) -> Tuple[List[HybridSearchHit], nx.DiGraph]:
        """Search and extract the reasoning subgraph for the top hits."""
        hits = self.search(query, k=k)
        hit_ids = [self.id_fn(hit.item) for hit in hits]
        subgraph = self.extract_reasoning_subgraph(hit_ids, max_depth=max_depth)
        return hits, subgraph

    def fast_path_search(self, query: str) -> Optional[Any]:
        hits = self.search(query, k=1)
        if not hits:
            return None
        return hits[0].item

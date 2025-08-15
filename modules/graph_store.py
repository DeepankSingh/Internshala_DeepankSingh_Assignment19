from typing import List, Dict, Any
import networkx as nx

class GraphStore:
    def __init__(self):
        self.G = nx.Graph()
        self.entity_to_chunks = {}

    def add_entity(self, entity: str):
        if entity and not self.G.has_node(entity):
            self.G.add_node(entity, type="entity")

    def add_relation(self, e1: str, e2: str, rel: str = "co_occurs"):
        if e1 and e2:
            self.add_entity(e1)
            self.add_entity(e2)
            if self.G.has_edge(e1, e2):
                self.G[e1][e2]["weight"] += 1.0
            else:
                self.G.add_edge(e1, e2, relation=rel, weight=1.0)

    def link_entity_to_chunk(self, entity: str, chunk_id: str):
        self.add_entity(entity)
        self.entity_to_chunks.setdefault(entity, set()).add(chunk_id)

    def related_entities(self, entity: str, k: int = 10) -> List[str]:
        if entity not in self.G:
            return []
        neighbors = sorted(self.G.neighbors(entity), key=lambda n: self.G[entity][n].get("weight", 1.0), reverse=True)
        return neighbors[:k]

    def expand_from_query_entities(self, entities: List[str], depth: int = 1, top_entities: int = 20) -> List[str]:
        seen = set()
        frontier = list(entities)
        result = set(entities)
        for _ in range(depth):
            next_frontier = []
            for e in frontier:
                for n in self.related_entities(e, k=top_entities):
                    if n not in seen:
                        result.add(n)
                        next_frontier.append(n)
                seen.add(e)
            frontier = next_frontier
        return list(result)

    def chunks_from_entities(self, entities: List[str], max_chunks: int = 20) -> List[str]:
        chunk_scores = {}
        for e in entities:
            for ch in self.entity_to_chunks.get(e, []):
                chunk_scores[ch] = chunk_scores.get(ch, 0) + 1.0
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in ranked[:max_chunks]]

    def to_pyvis(self):
        nodes = [{"id": n, "label": n, "title": f"type: {self.G.nodes[n].get('type','entity')}"} for n in self.G.nodes]
        edges = [{"from": u, "to": v, "title": self.G[u][v].get("relation","co_occurs"), "value": self.G[u][v].get("weight",1.0)} for u, v in self.G.edges]
        return nodes, edges

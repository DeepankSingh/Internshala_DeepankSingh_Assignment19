from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

class VectorStore:
    def __init__(self, persist_dir: str = "storage", collection_name: str = "docs",
                 model_name: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
        self.embedder = SentenceTransformer(model_name)

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name, metadata={"hnsw:space": "cosine"})

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        embeddings = self._embed(texts)
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        q_emb = self._embed([query_text])[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents","metadatas","distances","ids"])
        return res

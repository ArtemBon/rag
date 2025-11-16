from typing import List, Dict
import numpy as np
from processors.embedder import Embedder


class SemanticRetriever:
  """
  Retrieves relevant document chunks using semantic similarity.

  Uses cosine similarity between query and document embeddings
  to find the most relevant chunks.
  """

  def __init__(self, embedder: Embedder, top_k: int = 5):
    self.embedder = embedder
    self.top_k = top_k

  def retrieve(self, query: str, chunks: List[Dict[str, str]], embeddings: np.ndarray) -> str:
    query_embedding = self.embedder.embed_query(query)
    similarities = self._calculate_similarities(query_embedding, embeddings)
    top_indexes = np.argsort(similarities, axis=0)[-self.top_k:][::-1].tolist()
    context = self._build_context(chunks, top_indexes)
    return context

  def _calculate_similarities(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
    return np.dot(document_embeddings, query_embedding.T)

  def _build_context(self, chunks: List[Dict[str, str]], top_indexes: List[int]) -> str:
    context_parts = []
    for i, idx in enumerate(top_indexes, 1):
      chunk = chunks[idx]
      context_parts.append(f"[Source {i}: {chunk['title']}]")
      context_parts.append(chunk['content'])
      context_parts.append("")
    return "\n".join(context_parts)

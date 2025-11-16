from pathlib import Path
from typing import List, Optional
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
  """Generates embeddings with caching and lazy model loading."""

  def __init__(self, model_name: str, cache_file: str):
    self.model_name = model_name
    self.cache_file = cache_file
    self.model = None

  def embed_texts(self, texts: List[str]) -> np.ndarray:
    cached_embeddings = self._load_from_cache()
    if cached_embeddings is not None:
      return cached_embeddings

    print("Computing new embeddings...")
    self._load_model()

    embeddings = self.model.encode(
      texts,
      show_progress_bar=True,
      convert_to_numpy=True,
      normalize_embeddings=True
    )

    self._save_to_cache(embeddings)
    return embeddings

  def embed_query(self, query: str) -> np.ndarray:
    self._load_model()
    embedding = self.model.encode(
      query,
      convert_to_numpy=True,
      normalize_embeddings=True
    )
    return embedding

  def _load_model(self) -> None:
    if self.model is None:
      print(f"Loading embedding model: {self.model_name}")
      self.model = SentenceTransformer(self.model_name)
      print(f"Embedding model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}")

  def _load_from_cache(self) -> Optional[np.ndarray]:
    if Path(self.cache_file).exists():
      print(f"Loading embeddings from {self.cache_file}")
      with open(self.cache_file, 'r') as f:
        embeddings_list = json.load(f)
      return np.array(embeddings_list)
    return None

  def _save_to_cache(self, embeddings: np.ndarray) -> None:
    with open(self.cache_file, 'w') as f:
      f.write('[\n')
      embeddings_list = embeddings.tolist()
      for i, embedding in enumerate(embeddings_list):
        if i < len(embeddings_list) - 1:
          f.write(f'  {json.dumps(embedding)},\n')
        else:
          f.write(f'  {json.dumps(embedding)}\n')
      f.write(']\n')
    print(f"Saved embeddings to {self.cache_file}")

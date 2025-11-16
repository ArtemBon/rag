"""
Configuration settings for RAG system.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
  """Configuration constants for the RAG system."""

  # Data paths
  GITHUB_PATH = 'github'
  GITLAB_PATH = 'gitlab'

  # Chunking settings
  CHUNK_SIZE = 256
  CHUNK_OVERLAP = 25

  # Retrieval settings
  TOP_K = 5

  # Model settings
  EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  LLM_MODEL = "openrouter/gpt-oss-20b"

  # Embeddings cache settings
  EMBEDDINGS_CACHE_FILE = "embeddings_cache.json"

  # API settings
  OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

  # LLM settings
  TEMPERATURE = 0.1
  MAX_TOKENS = 3000

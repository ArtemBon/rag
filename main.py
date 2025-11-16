import argparse
from config import Config
from loaders.markdown_loader import MarkdownLoader
from loaders.erb_loader import ErbLoader
from processors.chunker import DocumentChunker
from processors.embedder import Embedder
from retrieval.retriever import SemanticRetriever
from llm.client import LLMClient


def parse_args():
  parser = argparse.ArgumentParser(description='RAG System - Question Answering with Document Retrieval')
  parser.add_argument('-t', '--text', type=str, required=True, help='User query text')
  return parser.parse_args()

args = parse_args()
question = args.text

md_loader = MarkdownLoader()
erb_loader = ErbLoader()
chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
embedder = Embedder(model_name=Config.EMBEDDING_MODEL, cache_file=Config.EMBEDDINGS_CACHE_FILE)
retriever = SemanticRetriever(embedder=embedder, top_k=Config.TOP_K)
llm_client = LLMClient(
  model_name=Config.LLM_MODEL,
  api_key=Config.OPENROUTER_API_KEY,
  temperature=Config.TEMPERATURE,
  max_tokens=Config.MAX_TOKENS
)

print("\n--- Loading documents ---")
md_documents = md_loader.load(Config.GITHUB_PATH)
erb_documents = erb_loader.load(Config.GITLAB_PATH)
documents = md_documents + erb_documents

print("\n--- Chunking documents ---")
chunks = chunker.chunk_documents(documents)

print("\n--- Generating embeddings ---")
chunk_texts = [chunk['content'] for chunk in chunks]
embeddings = embedder.embed_texts(chunk_texts)

print("\n--- Retrieving relevant context ---")
context = retriever.retrieve(question, chunks, embeddings)

print("\n--- Generating answer ---")
answer = llm_client.answer(question, context)

print("\n" + "="*60)
print("QUESTION:")
print(question)
print("\n" + "="*60)
print("ANSWER:")
print(answer)
print("="*60 + "\n")

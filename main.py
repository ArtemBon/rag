from pathlib import Path
from typing import List, Dict
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from litellm import completion
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_markdown(doc_path: str) -> List[Dict[str, str]]:
  """
  Load all markdown documents from specified path.

  Args:
    doc_path: Path to folder with markdown files

  Returns:
    List of documents with metadata:
    [
      {
        'content': 'document text',
        'source': 'path/to/file.md',
        'title': 'Article Title',
        'metadata': {...}
      },
      ...
    ]
  """
  documents = []
  base_path = Path(doc_path)
  md_files = list(base_path.rglob('index.md'))

  for md_file in md_files:
    with open(md_file, 'r', encoding='utf-8') as f:
      content = f.read()

    metadata = {}
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    frontmatter = frontmatter_match.group(1)
    for line in frontmatter.split('\n'):
      if ':' in line and not line.strip().startswith('#'):
        key, value = line.split(':', 1)
        metadata[key.strip()] = value.strip()

    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)

    # Clean markdown content
    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    # Remove excessive newlines
    content = re.sub(r'\n{3,}', '\n\n', content)
    # Remove image markdown but keep alt text
    content = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', content)
    # Keep links but extract text
    content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
    # Remove inline code backticks but keep content
    content = re.sub(r'`([^`]+)`', r'\1', content)
    # Remove code block markers but keep content
    content = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', content, flags=re.DOTALL)

    documents.append({
      'content': content.strip(),
      'source': str(md_file),
      'title': metadata.get('title', md_file.parent.name),
      'metadata': metadata
    })

  print(f"Loaded {len(documents)} markdown documents from {doc_path}")
  return documents

def load_erb(doc_path: str) -> List[Dict[str, str]]:
  """
  Load all ERB documents from specified path.

  Args:
    doc_path: Path to folder with ERB files

  Returns:
    List of documents with metadata:
    [
      {
        'content': 'document text',
        'source': 'path/to/file.html.erb',
        'title': 'Page Title',
        'metadata': {...}
      },
      ...
    ]
  """
  documents = []
  base_path = Path(doc_path)
  erb_files = list(base_path.rglob('*.erb'))

  for erb_file in erb_files:
    with open(erb_file, 'r', encoding='utf-8') as f:
      content = f.read()

    # Remove ERB code blocks (<% ... %> and <%= ... %>)
    content = re.sub(r'<%=?.*?%>', '', content, flags=re.DOTALL)

    # Remove HTML tags but keep text content
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<[^>]+>', ' ', content)

    # Remove HTML entities
    content = re.sub(r'&nbsp;', ' ', content)
    content = re.sub(r'&[a-zA-Z]+;', ' ', content)

    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Generate title from filename if not found in metadata
    title = erb_file.stem.replace('_', ' ').replace('.html', '').title()

    documents.append({
      'content': content.strip(),
      'source': str(erb_file),
      'title': title,
      'metadata': {}
    })

  print(f"Loaded {len(documents)} ERB documents from {doc_path}")
  return documents

def chunk_documents(documents: List[Dict[str, str]], chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict[str, str]]:
  """
  Chunk multiple documents into smaller pieces.

  Args:
    documents: List of documents with 'content', 'source', 'title', 'metadata'
    chunk_size: Maximum number of tokens (words) per chunk
    chunk_overlap: Number of tokens to overlap between chunks

  Returns:
    List of chunks with metadata:
    [
      {
        'content': 'chunk text',
        'source': 'path/to/file',
        'title': 'Document Title',
        'chunk_id': 0,
        'metadata': {...}
      },
      ...
    ]
  """
  all_chunks = []

  for doc in documents:
    text = doc['content']

    # Split text into sentences
    # Split by common sentence endings (., !, ?)
    sentences = re.split(r'[.!?]+\s+', text)
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Process sentences into chunks
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
      sentence_length = len(sentence.split())

      # If single sentence is longer than chunk_size, split it
      if sentence_length > chunk_size:
        # Save current chunk if exists
        if current_chunk:
          chunks.append(' '.join(current_chunk))
          current_chunk = []
          current_length = 0

        # Split long sentence into words
        words = sentence.split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
          chunk_words = words[i:i + chunk_size]
          chunks.append(' '.join(chunk_words))
      else:
        # Check if adding this sentence exceeds chunk_size
        if current_length + sentence_length > chunk_size:
          # Save current chunk
          chunks.append(' '.join(current_chunk))

          # Start new chunk with overlap
          overlap_words = ' '.join(current_chunk).split()[-chunk_overlap:]
          current_chunk = overlap_words + [sentence]
          current_length = len(overlap_words) + sentence_length
        else:
          # Add to current chunk
          current_chunk.append(sentence)
          current_length += sentence_length

    # Add last chunk
    if current_chunk:
      chunks.append(' '.join(current_chunk))

    # Create chunk documents with metadata
    for i, chunk_text in enumerate(chunks):
      chunk = {
        'content': chunk_text,
        'source': doc['source'],
        'title': doc['title'],
        'chunk_id': i,
        'metadata': doc.get('metadata', {})
      }
      all_chunks.append(chunk)

  print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
  return all_chunks

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", cache_file: str = "embeddings_cache.json") -> np.ndarray:
  """
  Generate embeddings for a list of texts using sentence-transformers.
  Automatically checks cache and saves results.

  Args:
    texts: List of text strings to embed
    model_name: Name of the sentence-transformers model to use
    cache_file: Path to cache file (.json format)

  Returns:
    Numpy array of embeddings with shape (len(texts), embedding_dim)
    Embeddings are normalized for cosine similarity
  """
  # Try to load from cache
  if Path(cache_file).exists():
    print(f"Loading embeddings from {cache_file}")
    with open(cache_file, 'r') as f:
      embeddings_list = json.load(f)
    embeddings = np.array(embeddings_list)
    return embeddings

  # Cache miss - generate new embeddings
  print(f"Computing new embeddings...")
  print(f"Loading embedding model: {model_name}")
  model = SentenceTransformer(model_name)
  print(f"Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}")

  # Generate embeddings
  embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
  )

  # Save to cache as JSON with custom formatting
  with open(cache_file, 'w') as f:
    f.write('[\n')
    embeddings_list = embeddings.tolist()
    for i, embedding in enumerate(embeddings_list):
      if i < len(embeddings_list) - 1:
        f.write(f'  {json.dumps(embedding)},\n')
      else:
        f.write(f'  {json.dumps(embedding)}\n')
    f.write(']\n')
  print(f"Saved embeddings to {cache_file}")

  return embeddings

def embed_query(query: str, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2") -> np.ndarray:
  """
  Generate embedding for a single query.

  Args:
    query: Query text

  Returns:
    Numpy array of embedding with shape (embedding_dim,)
  """
  # Load the sentence-transformers model
  print(f"Loading embedding model: {model_name}")
  model = SentenceTransformer(model_name)
  print(f"Embedding model loaded. Dimension: {model.get_sentence_embedding_dimension()}")

  embedding = model.encode(
    query,
    convert_to_numpy=True,
    normalize_embeddings=True
  )
  return embedding

def build_context(chunks: List[Dict[str, str]], top_indexes: List[int]) -> str:
  """
  Build context string from retrieved chunks.

  Args:
    results: List of retrieved chunks with structure:
      [
        {
          'content': 'chunk text',
          'title': 'Document Title',
          'source': 'path/to/file'
          ...
        },
        ...
      ]

  Returns:
    Formatted context string with sources and content
  """
  context_parts = []

  for i, idx in enumerate(top_indexes, 1):
    chunk = chunks[idx]
    # Add source header
    context_parts.append(f"[Source {i}: {chunk['title']}]")
    # Add content
    context_parts.append(chunk['content'])
    # Add empty line separator
    context_parts.append("")

  return "\n".join(context_parts)

documents = load_markdown('github') + load_erb('gitlab')
chunks = chunk_documents(documents)

chunk_texts = [chunk['content'] for chunk in chunks]
embeddings = embed_texts(chunk_texts)

question = "What is an MVP and how to build one effectively?"
embedding = embed_query(question)

similarities = np.dot(embeddings, embedding.T)
print(similarities)

top_5_indexes = np.argsort(similarities, axis=0)[-5:][::-1].tolist()
print(top_5_indexes)

context = build_context(chunks, top_5_indexes)

system_prompt = """
  You are a knowledgeable expert assistant.

  Instructions:
  1. Answer questions using ONLY information from the context provided below
  2. You can rephrase and organize information from the context into a coherent answer
  3. DO NOT add any factual claims, statistics, technical details, examples, or specific information that is not explicitly stated in the context
  4. You may use natural connecting words and phrases to structure the answer (like "additionally", "however", "furthermore")
  5. Present the information naturally as your direct expert knowledge
  6. Provide detailed answers using ALL relevant information found in the context
  7. For multi-part questions, address each part using information from the context
  8. Respond in the same language as the question

  CRITICAL: NEVER use these phrases or similar ones:
  - "the context"
  - "the sources"
  - "according to"
  - "based on the information provided"
  - "the information describes"
  - "it is mentioned that"
  - "as stated"
  - "the document says"

  Write as if you directly know this information, not as if you're reading it from somewhere.
"""

user_prompt = f"""
  Context:
  {context}

  Question: {question}

  Answer:
"""


print("\n\n\n\n===================================================\n")
print(context)
print("\n===================================================\n\n\n\n")


response = completion(
  model=f"openrouter/gpt-oss-20b",
  messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
  ],
  api_key=os.getenv("OPENROUTER_API_KEY"),
  temperature=0.1,
  max_tokens=3000
)

answer = response.choices[0].message.content.strip()
print(answer)

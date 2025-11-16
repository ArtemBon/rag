from typing import List, Dict
import re


class DocumentChunker:
  """Chunks documents into smaller pieces with overlap."""

  def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

  def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    all_chunks = []

    for doc in documents:
      text = doc['content']
      sentences = self._split_into_sentences(text)
      chunks = self._create_chunks(sentences)

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

  def _split_into_sentences(self, text: str) -> List[str]:
    sentences = re.split(r'[.!?]+\s+', text)
    return [s.strip() for s in sentences if s.strip()]

  def _create_chunks(self, sentences: List[str]) -> List[str]:
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
      sentence_length = len(sentence.split())

      if sentence_length > self.chunk_size:
        if current_chunk:
          chunks.append(' '.join(current_chunk))
          current_chunk = []
          current_length = 0

        words = sentence.split()
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
          chunk_words = words[i:i + self.chunk_size]
          chunks.append(' '.join(chunk_words))
      else:
        if current_length + sentence_length > self.chunk_size:
          chunks.append(' '.join(current_chunk))
          overlap_words = ' '.join(current_chunk).split()[-self.chunk_overlap:]
          current_chunk = overlap_words + [sentence]
          current_length = len(overlap_words) + sentence_length
        else:
          current_chunk.append(sentence)
          current_length += sentence_length

    if current_chunk:
      chunks.append(' '.join(current_chunk))

    return chunks

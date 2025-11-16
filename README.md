# RAG System

The system follows these steps while processing the question:

```
Documents → Loaders → Chunker → Embedder → Retriever → LLM → Answer
```

### Components

- **Loaders**: Extract content from different document formats

  - `MarkdownLoader`: Processes Markdown files with YAML frontmatter
  - `ErbLoader`: Handles ERB template files

- **Processors**: Transform documents for efficient retrieval

  - `DocumentChunker`: Splits documents into semantic chunks with overlap
  - `Embedder`: Generates vector embeddings with caching support

- **Retrieval**: Find relevant content for queries

  - `SemanticRetriever`: Uses cosine similarity for semantic search

- **LLM**: Generate natural language answers
  - `LLMClient`: Interfaces with language models via LiteLLM

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd rag
```

2. Install dependencies:

```bash
pip install sentence-transformers litellm python-dotenv numpy
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

4. Prepare your documents:

- Place Markdown documents in the `github/` directory
- Place ERB templates in the `gitlab/` directory

## Usage

### Basic Query

Run a query against your document collection:

```bash
python main.py -t "Your question here"
```

## How it works

The system operates as a pipeline, processing documents through several stages:

**Documents → Loaders → Chunker → Embedder → Retriever → LLM → Answer**

The **loaders** read files from the filesystem. The MarkdownLoader processes Markdown files and extracts YAML frontmatter if present. The ErbLoader handles ERB templates by removing Ruby code and HTML tags, leaving only the text content.

The **chunker** splits documents into smaller pieces of approximately 512 words each. The chunks overlap by about 50 words to avoid cutting sentences or ideas in half. The chunker works intelligently by splitting on sentence boundaries rather than arbitrary positions.

The **embedder** converts each text chunk into a vector - essentially a list of numbers that represents the semantic meaning of that text. It uses a multilingual sentence transformer model (paraphrase-multilingual-mpnet-base-v2).

Once the embeddings are calculated, they're saved to a JSON file called `embeddings_cache.json`. On subsequent runs, the system loads these cached embeddings instead of recalculating them, which saves significant time.

When a question is asked, the **retriever** converts the question into the same type of vector and compares it with all document chunks. The comparison uses cosine similarity - a mathematical method that measures how similar two vectors are by comparing the angles between them. Vectors pointing in similar directions indicate similar semantic meaning. The system selects the top 5 most similar chunks based on this measurement.

Finally, these 5 relevant chunks are sent to the LLM along with the question. The system instructs the AI to answer based only on the provided context. The AI then generates a response in the same language as the question.

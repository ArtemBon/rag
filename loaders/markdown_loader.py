from pathlib import Path
from typing import List, Dict
import re
from loaders.base_loader import BaseLoader


class MarkdownLoader(BaseLoader):
  """Loader for markdown documents with YAML frontmatter."""

  def load(self, path: str) -> List[Dict[str, str]]:
    documents = []
    base_path = Path(path)
    md_files = list(base_path.rglob('index.md'))

    for md_file in md_files:
      with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

      metadata = self._extract_frontmatter(content)
      content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
      content = self._clean_markdown(content)

      documents.append({
        'content': content.strip(),
        'source': str(md_file),
        'title': metadata.get('title', md_file.parent.name),
        'metadata': metadata
      })

    print(f"Loaded {len(documents)} markdown documents from {path}")
    return documents

  def _extract_frontmatter(self, content: str) -> Dict[str, str]:
    metadata = {}
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)

    if frontmatter_match:
      frontmatter = frontmatter_match.group(1)
      for line in frontmatter.split('\n'):
        if ':' in line and not line.strip().startswith('#'):
          key, value = line.split(':', 1)
          metadata[key.strip()] = value.strip()

    return metadata

  def _clean_markdown(self, content: str) -> str:
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r'!\[(.*?)\]\(.*?\)', r'\1', content)
    content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
    content = re.sub(r'`([^`]+)`', r'\1', content)
    content = re.sub(r'```[\w]*\n(.*?)\n```', r'\1', content, flags=re.DOTALL)
    return content

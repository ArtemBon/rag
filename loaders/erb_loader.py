from pathlib import Path
from typing import List, Dict
import re
from loaders.base_loader import BaseLoader


class ErbLoader(BaseLoader):
  """Loader for ERB (Embedded Ruby) template files."""

  def load(self, path: str) -> List[Dict[str, str]]:
    documents = []
    base_path = Path(path)
    erb_files = list(base_path.rglob('*.erb'))

    for erb_file in erb_files:
      with open(erb_file, 'r', encoding='utf-8') as f:
        content = f.read()

      content = self._remove_erb_code(content)
      content = self._strip_html(content)
      title = self._generate_title(erb_file.stem)

      documents.append({
        'content': content.strip(),
        'source': str(erb_file),
        'title': title,
        'metadata': {}
      })

    print(f"Loaded {len(documents)} ERB documents from {path}")
    return documents

  def _remove_erb_code(self, content: str) -> str:
    return re.sub(r'<%=?.*?%>', '', content, flags=re.DOTALL)

  def _strip_html(self, content: str) -> str:
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<[^>]+>', ' ', content)
    content = re.sub(r'&nbsp;', ' ', content)
    content = re.sub(r'&[a-zA-Z]+;', ' ', content)
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content

  def _generate_title(self, filename: str) -> str:
    return filename.replace('_', ' ').replace('.html', '').title()

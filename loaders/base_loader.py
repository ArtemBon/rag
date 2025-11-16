from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLoader(ABC):
  """Abstract base class for document loaders."""

  @abstractmethod
  def load(self, path: str) -> List[Dict[str, str]]:
    pass
"""Base protocol for document loaders."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from app.domain.models import RetrievedDoc


class DocumentLoader(ABC):
    @abstractmethod
    def load(self) -> Sequence[RetrievedDoc]:
        """Return documents ready for embedding + upsert."""
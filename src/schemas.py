from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class IngestResult:
    doc_hash: str
    status: Literal["indexed", "cached"]
    chunk_count: int
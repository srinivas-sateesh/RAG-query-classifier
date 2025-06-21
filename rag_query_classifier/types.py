from enum import Enum
from dataclasses import dataclass
from typing import Optional

class QueryLabel(Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    VAGUE = "vague"
    UNKNOWN = "unknown"

@dataclass
class ClassificationResult:
    label: QueryLabel
    confidence_score: float
    reasoning: str
    source: str = "unknown"
from enum import Enum

class QueryLabel(Enum):
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    VAGUE = "vague"
    UNKNOWN = "unknown"
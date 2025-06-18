import re
from .types import QueryLabel

class RuleBasedClassifier:
    def __init__(self, rules):
        self.rules = rules

    def classify(self, query: str):
        q = query.lower()
        for kw in self.rules.get("relevant_keywords", []):
            if kw in q:
                return QueryLabel.RELEVANT
        for pat in self.rules.get("irrelevant_patterns", []):
            if re.search(pat, q):
                return QueryLabel.IRRELEVANT
        for pat in self.rules.get("vague_patterns", []):
            if re.search(pat, q):
                return QueryLabel.VAGUE
        return None
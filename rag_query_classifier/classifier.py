from .config import Config
from .rules import RuleBasedClassifier
from .llm import LLMClassifier
from .actions import action_for
from .types import QueryLabel, ClassificationResult
import re


class QueryClassifier:
    def __init__(self, config_path="config.yaml", model="phi3:mini"):
        self.config = Config(config_path)
        self.rule_classifier = RuleBasedClassifier(self.config.rules)
        self.llm_classifier = LLMClassifier(examples=self.config.examples, model=model)
        
        # Cache for repeated queries
        self._query_cache = {}

    def _quick_classify(self, query: str) -> ClassificationResult:
        """Fast classification using generic patterns only"""
        query_lower = query.lower()
        
        # Check for very short queries (vague)
        if len(query.strip()) < 5:
            return ClassificationResult(
                label=QueryLabel.VAGUE,
                confidence_score=0.7,
                reasoning="Query too short to be meaningful",
                source="quick_pattern"
            )
        
        return None

    def classify(self, query: str) -> ClassificationResult:
        # Check cache first
        if query in self._query_cache:
            cached_result = self._query_cache[query]
            cached_result.source = "cache"
            return cached_result
        
        # Try quick classification first (fastest)
        quick_result = self._quick_classify(query)
        if quick_result:
            self._query_cache[query] = quick_result
            return quick_result
        
        # Try rule-based classification
        rule_label = self.rule_classifier.classify(query)
        if rule_label:
            result = ClassificationResult(
                label=rule_label,
                confidence_score=1.0,  # Rules are deterministic
                reasoning="Matched by rule-based classifier",
                source="rule"
            )
            self._query_cache[query] = result
            return result
        
        # Fall back to LLM classification
        try:
            result = self.llm_classifier.classify(query)
            self._query_cache[query] = result
            return result
        except Exception as e:
            print(f"LLM classification failed: {e}")
            result = ClassificationResult(
                label=QueryLabel.UNKNOWN,
                confidence_score=0.0,
                reasoning=f"LLM classification failed: {e}",
                source="llm (failed)"
            )
            self._query_cache[query] = result
            return result

    def action(self, result: ClassificationResult) -> str:
        return action_for(result.label)
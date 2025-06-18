from .config import Config
from .rules import RuleBasedClassifier
from .llm import LLMClassifier
from .actions import action_for
from .types import QueryLabel

class QueryClassifier:
    def __init__(self, config_path="config.yaml"):
        self.config = Config(config_path)
        self.rule_classifier = RuleBasedClassifier(self.config.rules)
        self.llm_classifier = LLMClassifier(examples=self.config.examples)

    def classify(self, query: str):
        label = self.rule_classifier.classify(query)
        if label:
            return label, "rule"
        try:
            label = self.llm_classifier.classify(query)
            return label, "llm"
        except Exception as e:
            print(f"LLM classification failed: {e}")
            return QueryLabel.UNKNOWN, "llm (failed)"


    def action(self, label):
        return action_for(label)
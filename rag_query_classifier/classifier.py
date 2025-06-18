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

    def classify(self, query: str) -> QueryLabel:
        label = self.rule_classifier.classify(query)
        if label:
            return label
        return self.llm_classifier.classify(query)

    def action(self, label: QueryLabel) -> str:
        return action_for(label)
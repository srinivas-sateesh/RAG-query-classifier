import requests
from .types import QueryLabel
from .exceptions import LLMError

class LLMClassifier:
    def __init__(self, endpoint="http://localhost:11434/api/generate", model="llama3", examples=None):
        self.endpoint = endpoint
        self.model = model
        self.examples = examples or {}

    def build_prompt(self, query: str):
        prompt = "Classify the following query as 'relevant', 'irrelevant', or 'vague'.\n"
        # Add few-shot examples from config
        for label, exs in self.examples.items():
            for ex in exs:
                prompt += f"Query: {ex}\nLabel: {label}\n"
        prompt += f"Query: {query}\nLabel:"
        return prompt

    def classify(self, query: str):
        prompt = self.build_prompt(query)
        try:
            response = requests.post(
                self.endpoint,
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=5,
            )
            result = response.json()["response"].strip().lower()
            if "relevant" in result:
                return QueryLabel.RELEVANT
            if "irrelevant" in result:
                return QueryLabel.IRRELEVANT
            if "vague" in result:
                return QueryLabel.VAGUE
            return QueryLabel.UNKNOWN
        except Exception as e:
            raise LLMError(f"LLM classification failed: {e}")
import requests
from .types import QueryLabel
from .exceptions import LLMError



class LLMClassifier:
    def __init__(self, endpoint="http://localhost:11434/api/generate", model="phi3:mini", examples=None):
        self.endpoint = endpoint
        self.model = model
        self.examples = examples or {}

    def build_system_prompt(self):
        return (
            "You are a query classifier. Respond with only one word. Do not give any reasoning. Just one word only.Only use one of these labels: 'relevant', 'irrelevant', 'vague'.\n"
            "Classify queries based on the following examples.\n"
            "Business queries are relevant."
            "Also label it as vague if queries have non-English words"
        )

    def build_examples(self):
        prompt = ""
        for label, exs in self.examples.items():
            for ex in exs:
                prompt += (
                    f"<example>\n"
                    f"<query>{ex}</query>\n"
                    f"<label>{label}</label>\n"
                    f"</example>\n"
                )
        return prompt

    def build_user_prompt(self, query: str):
        return f"<classify>\n<query>{query}</query>\n</classify>\n"

    def classify(self, query: str):
        system_prompt = self.build_system_prompt()
        examples_prompt = self.build_examples()
        user_prompt = self.build_user_prompt(query)
        full_prompt = system_prompt + examples_prompt + user_prompt

        response = requests.post(
            self.endpoint,
            json={"model": self.model, "prompt": full_prompt, "stream": False},
            timeout=10,
        )
        #result = response.json()["response"].strip().lower()
        result = response.json()["response"]
        print("*=50")
        print(result)
        # Extract label from response
     
        return self.extract_label(result) 

    def extract_label(self, response: str):
        response = response.strip().lower()
    # Look for exact matches first
        if " relevant" in response:
            return "relevant"
        elif "irrelevant" in response:
            return "irrelevant" 
        elif "vague" in response:
            return "vague"
        else:
            return "vague"  # default fallback
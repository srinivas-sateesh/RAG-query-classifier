import requests
import json
from .types import QueryLabel, ClassificationResult
from .exceptions import LLMError


class LLMClassifier:
    def __init__(self, endpoint="http://localhost:11434/api/generate", model="phi3:instruct", examples=None, cache_size=1000):
        self.endpoint = endpoint
        self.model = model
        self.examples = examples or {}
        self.cache_size = cache_size
        
        # Cache the system prompt and examples prompt
        self._system_prompt = None
        self._examples_prompt = None
        
        # Use session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Simple LRU cache for results
        self._result_cache = {}
        self._cache_keys = []

    def _get_system_prompt(self):
        if self._system_prompt is None:
            self._system_prompt = """You are a query classifier. You must respond ONLY with valid JSON in the following format:
{
  "label": "relevant|irrelevant|vague",
  "confidence_score": 0.0-1.0,
  "reasoning": "brief explanation of classification"
}

Rules:
- Queries with non-English words are vague
- Only use the exact labels: 'relevant', 'irrelevant', 'vague'
- Confidence score should be between 0.0 and 1.0
- Provide brief reasoning for your classification
- Classify based on query clarity and relevance to any domain
"""
        return self._system_prompt

    def _get_examples_prompt(self):
        if self._examples_prompt is None:
            if not self.examples:
                self._examples_prompt = ""
                return self._examples_prompt

            example_prompts = []
            for label, exs in self.examples.items():
                for ex in exs:
                    mock_response = {
                        "label": label,
                        "confidence_score": 0.9,
                        "reasoning": f"Example of a {label} query."
                    }
                    example_prompts.append(f'Query: "{ex}"\nJSON Response: {json.dumps(mock_response)}')

            if example_prompts:
                self._examples_prompt = "Here are some examples:\n\n" + "\n\n".join(example_prompts) + "\n\n"
            else:
                self._examples_prompt = ""
        return self._examples_prompt

    def _cache_result(self, query: str, result: ClassificationResult):
        """Simple LRU cache implementation"""
        if query in self._result_cache:
            # Move to end (most recently used)
            self._cache_keys.remove(query)
        else:
            # Remove oldest if cache is full
            if len(self._cache_keys) >= self.cache_size:
                oldest = self._cache_keys.pop(0)
                del self._result_cache[oldest]
        
        self._result_cache[query] = result
        self._cache_keys.append(query)

    def build_user_prompt(self, query: str):
        return f'Query: "{query}"\nJSON Response:'

    def classify(self, query: str) -> ClassificationResult:
        # Check cache first
        if query in self._result_cache:
            cached_result = self._result_cache[query]
            # Update source to indicate it's from cache
            cached_result.source = f"llm ({self.model}) cached"
            return cached_result
        
        # Use cached prompts
        system_prompt = self._get_system_prompt()
        examples_prompt = self._get_examples_prompt()
        user_prompt = self.build_user_prompt(query)
        full_user_prompt = examples_prompt + user_prompt

        payload = {
            "model": self.model,
            "prompt": full_user_prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        response = self.session.post(
            self.endpoint,
            json=payload,
            timeout=10,
        )
        
        result = response.json()["response"]
        print("*=50")
        print(result)
        
        classification_result = self.parse_json_response(result)
        
        # Cache the result
        self._cache_result(query, classification_result)
        
        return classification_result

    def parse_json_response(self, response: str) -> ClassificationResult:
        try:
            # Clean the response - remove any leading/trailing whitespace and common prefixes
            cleaned_response = response.strip()
            
            # Try to find JSON in the response if it's not pure JSON
            if not cleaned_response.startswith('{'):
                # Look for JSON object in the response
                start_idx = cleaned_response.find('{')
                end_idx = cleaned_response.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    cleaned_response = cleaned_response[start_idx:end_idx+1]
            
            # Try to parse as JSON
            response_data = json.loads(cleaned_response)
            
            # Extract fields from JSON
            label_str = response_data.get("label", "").lower()
            confidence_score = response_data.get("confidence_score", 0.0)
            reasoning = response_data.get("reasoning", "")
            
            # Map string label to QueryLabel enum
            label = self.map_label_to_enum(label_str)
            
            return ClassificationResult(
                label=label,
                confidence_score=confidence_score,
                reasoning=reasoning,
                source=f"llm ({self.model})"
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Try to extract JSON from the response more aggressively
            try:
                # Look for JSON pattern more carefully
                import re
                json_pattern = r'\{[^{}]*"label"[^{}]*"confidence_score"[^{}]*"reasoning"[^{}]*\}'
                json_match = re.search(json_pattern, response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    response_data = json.loads(json_str)
                    
                    label_str = response_data.get("label", "").lower()
                    confidence_score = response_data.get("confidence_score", 0.0)
                    reasoning = response_data.get("reasoning", "")
                    
                    label = self.map_label_to_enum(label_str)
                    
                    return ClassificationResult(
                        label=label,
                        confidence_score=confidence_score,
                        reasoning=reasoning,
                        source=f"llm ({self.model})"
                    )
            except:
                pass
            
            # Fallback to old parsing method if JSON parsing fails
            print(f"JSON parsing failed: {e}, falling back to text parsing")
            print(f"Raw response: {repr(response)}")
            label = self.extract_label(response)
            return ClassificationResult(
                label=label,
                confidence_score=0.5,  # Default confidence for fallback
                reasoning="Parsed from text response due to JSON parsing failure",
                source=f"llm ({self.model}) fallback"
            )

    def map_label_to_enum(self, label_str: str) -> QueryLabel:
        """Map string label to QueryLabel enum"""
        label_mapping = {
            "relevant": QueryLabel.RELEVANT,
            "irrelevant": QueryLabel.IRRELEVANT,
            "vague": QueryLabel.VAGUE
        }
        return label_mapping.get(label_str, QueryLabel.UNKNOWN)

    def extract_label(self, response: str):
        response = response.strip().lower()
        # Look for exact matches first
        if " relevant" in response:
            return QueryLabel.RELEVANT
        elif "irrelevant" in response:
            return QueryLabel.IRRELEVANT 
        elif "vague" in response:
            return QueryLabel.VAGUE
        else:
            return QueryLabel.UNKNOWN  # default fallback
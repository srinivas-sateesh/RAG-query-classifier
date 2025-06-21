import time
from rag_query_classifier import QueryClassifier

# Use phi3:mini for reliable classification (default)
# Other options: "tinyllama:latest", "llama2:7b", "mistral:7b", etc.
qc = QueryClassifier("config.yaml", model="phi3:mini")

# Clear any existing cache to test fresh classifications
if hasattr(qc, '_query_cache'):
    qc._query_cache.clear()

# Test queries (domain agnostic)
queries = [
    "Is my business prjhih?",  # Should be vague (non-English)
    "How do I solve this problem?",  # Should be relevant (clear question)
    "What's the weather like?",  # Should be relevant (clear question)
    "Is my business prjhih?",  # Should be cached (same as first)
    "How do I fix this?",  # Should be relevant (clear question)
    "xyz",  # Should be vague (too short)
    "What is the meaning of life?",  # Should be relevant (philosophical but clear)
    "asdfgh",  # Should be vague (gibberish)
]

print("Testing phi3:mini classifier performance...\n")

for i, query in enumerate(queries, 1):
    print(f"Query {i}: {query}")
    
    # Time the classification
    start_time = time.time()
    result = qc.classify(query)
    end_time = time.time()
    
    classification_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    print(f"  Classification: {result.label.value}")
    print(f"  Confidence Score: {result.confidence_score:.3f}")
    print(f"  Reasoning: {result.reasoning}")
    print(f"  Classified by: {result.source}")
    print(f"  Time taken: {classification_time:.2f} ms")
    print(f"  Action: {qc.action(result)}")
    print()
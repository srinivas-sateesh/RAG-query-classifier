import time
from rag_query_classifier import QueryClassifier

# Use phi3:mini for reliable classification (default)
# Other options: "tinyllama:latest", "llama2:7b", "mistral:7b", etc.
qc = QueryClassifier("config.yaml", model="phi3:mini")

# Clear any existing cache to test fresh classifications
if hasattr(qc, '_query_cache'):
    qc._query_cache.clear()

# Test queries (domain-specific for liver disease, with a mix of rule-based and LLM-based cases)
queries = [
    # Rule-based classifications
    "What are the symptoms of liver cirrhosis?",  # Relevant (keyword: "liver cirrhosis")
    "What's the weather like in New York?",      # Irrelevant (keyword: "weather")
    "I need help",                               # Vague (pattern from config)
    "Tell me about fatty liver disease",         # Relevant (keyword: "fatty liver disease")
    "Who won the last football game?",           # Irrelevant (keyword: "football")

    # LLM-based classifications
    "My skin and eyes are turning yellow.",      # Relevant (symptom of jaundice, no keyword)
    "Is it safe to drink alcohol every day?",    # Relevant (related to liver health)
    "What is the capital of Mongolia?",          # Irrelevant (general knowledge)
    "My system isn't functioning as expected.",  # Vague (no specific pattern)
    "Is it safe to drink alcohol every day?",    # Cached (same as a previous query)
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
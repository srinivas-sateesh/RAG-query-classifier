from rag_query_classifier import QueryClassifier, QueryLabel

# Initialize the classifier (adjust config path if needed)
qc = QueryClassifier("rag_query_classifier/config.yaml")

# Classify a query
query = "Can you help me with my invoice?"
label = qc.classify(query)

print(f"Classification: {label.value}")  # e.g., "relevant"
print(f"Action: {qc.action(label)}")     # e.g., "Proceed to LLM answer."
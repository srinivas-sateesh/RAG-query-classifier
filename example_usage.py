from rag_query_classifier import QueryClassifier

qc = QueryClassifier("config.yaml")
query = "Is my business prjhih?"
label, component = qc.classify(query)
print(query)
print(f"Classification: {label}")
print(f"Classified by: {component}")
print(f"Action: {qc.action(label)}")
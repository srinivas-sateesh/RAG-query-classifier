# RAG Query Classifier

A fast, domain-agnostic query classification system that determines whether user queries are **relevant**, **irrelevant**, or **vague**. Perfect for RAG (Retrieval-Augmented Generation) systems to filter and route queries appropriately.

## 🚀 Features

- **⚡ Fast Classification**: Multi-level classification pipeline with caching
- **🌐 Domain Agnostic**: Works for any domain (business, medical, legal, etc.)
- **🔧 Configurable**: Easy domain-specific customization via YAML config
- **🤖 LLM Integration**: Uses local LLMs via Ollama for complex classification
- **📊 Structured Output**: Returns confidence scores and reasoning
- **💾 Smart Caching**: LRU cache for instant repeated query responses
- **⚙️ Rule-Based Fallback**: Fast pattern matching before LLM calls

## 📋 Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) with a compatible model (e.g., `phi3:mini`, `llama2:7b`, `mistral:7b`)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd RAG-query-classifier
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate

   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install and start Ollama:**
   ```bash
   # Install Ollama (macOS/Linux)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull a model (recommended: phi3:mini)
   ollama pull phi3:mini
   ```

5. **Start Ollama server:**
   ```bash
   ollama serve
   ```

### Virtual Environment Management

**To deactivate the virtual environment:**
```bash
deactivate
```

**To reactivate the virtual environment later:**
```bash
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

**To remove the virtual environment:**
```bash
rm -rf venv
```

## ⚙️ Configuration

The classifier is configured via `config.yaml`. Here's the structure:

```yaml
rules:
  relevant_keywords:
    - "account"
    - "invoice"
    - "payment"
    - "order"
    - "subscription"
    - "reset password"
    - "password"
  irrelevant_patterns:
    - "weather"
    - "sports"
    - "joke"
    - "news"
    - "movie"
  vague_patterns:
    - "help"
    - "question"
    - "info"
    - "support"
    - "assist"

examples:
  relevant:
    - "How do I reset my account password?"
    - "What is the status of my last order?"
    - "Can I get a copy of my invoice?"
  irrelevant:
    - "Tell me a joke."
    - "What's the weather today?"
    - "Who won the football game?"
  vague:
    - "I need help."
    - "Can you assist me?"
    - "Support."
```

### Configuration Options

- **`relevant_keywords`**: Words/phrases that indicate relevant queries
- **`irrelevant_patterns`**: Patterns that indicate irrelevant queries
- **`vague_patterns`**: Patterns that indicate vague/unclear queries
- **`examples`**: Training examples for the LLM classifier

## 🚀 Quick Start

### Basic Usage

```python
from rag_query_classifier import QueryClassifier

# Initialize classifier
classifier = QueryClassifier("config.yaml")

# Classify a query
result = classifier.classify("How do I reset my password?")

print(f"Label: {result.label.value}")
print(f"Confidence: {result.confidence_score}")
print(f"Reasoning: {result.reasoning}")
print(f"Source: {result.source}")
print(f"Action: {classifier.action(result)}")
```

### Performance Testing

```python
import time
from rag_query_classifier import QueryClassifier

classifier = QueryClassifier("config.yaml")

queries = [
    "How do I change my password?",
    "What's the weather like?",
    "I need help with my account",
    "Tell me a joke",
    "asdfgh"  # gibberish
]

for query in queries:
    start_time = time.time()
    result = classifier.classify(query)
    end_time = time.time()
    
    print(f"Query: {query}")
    print(f"  Classification: {result.label.value}")
    print(f"  Confidence: {result.confidence_score:.3f}")
    print(f"  Time: {(end_time - start_time) * 1000:.2f} ms")
    print(f"  Source: {result.source}")
    print()
```

## 📊 Classification Results

The classifier returns a `ClassificationResult` object with:

- **`label`**: `QueryLabel.RELEVANT`, `QueryLabel.IRRELEVANT`, or `QueryLabel.VAGUE`
- **`confidence_score`**: Float between 0.0 and 1.0
- **`reasoning`**: String explanation of the classification
- **`source`**: Classification method used (`rule`, `llm (phi3:mini)`, `cache`, etc.)

## 🎯 Classification Logic

The classifier uses a multi-level approach:

1. **Cache Check** (1ms): Returns cached results instantly
2. **Quick Patterns** (1-5ms): Detects very short queries
3. **Rule-Based** (5-10ms): Matches config patterns
4. **LLM Classification** (1000-3000ms): Complex reasoning with local LLM

### Classification Criteria

- **Relevant**: Clear, specific questions related to your domain
- **Irrelevant**: Entertainment, weather, sports, etc.
- **Vague**: Too short, unclear, or contains non-English text

## 🏗️ Design Decisions

### Why Rule-Based + LLM Instead of Embeddings?

The classifier intentionally uses a **rule-based + LLM** approach rather than embedding-based classification. Here's why:

#### **Embedding-Based Approach (Not Used)**
```python
# What we could have done:
embeddings = embed_query(query)
similarity_scores = cosine_similarity(embeddings, example_embeddings)
classification = argmax(similarity_scores)
```

**Problems with embedding-based classification:**
- **🔧 Configuration Dependency**: Requires carefully curated example embeddings
- **📊 Training Data Needs**: Needs many high-quality examples per category
- **🎯 Threshold Tuning**: Requires manual similarity threshold tuning
- **⚡ Performance**: Embedding generation adds latency
- **🎲 Inconsistent Results**: Similarity scores can be unpredictable

#### **Rule-Based + LLM Approach (Current)**
```python
# What we actually do:
if rule_matches(query):
    return rule_classification
else:
    return llm_classification(query)
```

**Benefits of our approach:**
- **🚀 Fast Rules**: Instant classification for common patterns
- **🧠 Intelligent LLM**: Handles complex, nuanced cases
- **🔧 Simple Configuration**: Just keywords and patterns
- **⚡ Performance**: Rules are fast, LLM only when needed
- **🎯 Predictable**: Clear, deterministic rule matching
- **🔄 Low Maintenance**: Easy to update rules and examples

#### **Trade-offs**

| Aspect | Rule-Based + LLM | Embedding-Based |
|--------|------------------|-----------------|
| **Speed** | ⚡ Fast rules, slow LLM | 🐌 Embedding generation |
| **Accuracy** | 🎯 High (LLM reasoning) | 📊 Variable (similarity) |
| **Configuration** | 🔧 Simple keywords | 📝 Complex examples |
| **Predictability** | ✅ Deterministic | ❓ Unpredictable |
| **Domain Adaptation** | 🎯 Easy | 🔄 Requires retraining |

### **When to Use Each Approach**

**Use Rule-Based + LLM (Current) when:**
- You want fast, predictable classification
- You have clear domain rules and patterns
- You want low maintenance overhead
- You need explainable reasoning
- You have limited training data

**Consider Embedding-Based when:**
- You have extensive, high-quality training data
- You need semantic similarity matching
- You're willing to tune thresholds carefully
- You have the resources for embedding generation
- You need to handle many similar examples

Our approach prioritizes **simplicity**, **speed**, and **maintainability** over the complexity of embedding-based systems.

## 🔧 Advanced Configuration

### Using Different Models

```python
# Fast model (less accurate)
classifier = QueryClassifier("config.yaml", model="tinyllama:latest")

# Balanced model (default)
classifier = QueryClassifier("config.yaml", model="phi3:mini")

# Accurate model (slower)
classifier = QueryClassifier("config.yaml", model="llama2:7b")
```

### Custom Endpoint

```python
from rag_query_classifier import LLMClassifier

# Use custom Ollama endpoint
llm = LLMClassifier(
    endpoint="http://localhost:11434/api/generate",
    model="phi3:mini"
)
```

### Cache Configuration

```python
# Increase cache size
llm = LLMClassifier(cache_size=5000)

# Disable caching
llm = LLMClassifier(cache_size=0)
```

## 🏗️ Architecture

```
Query Input
    ↓
Cache Check → [Cached?] → Return Result
    ↓ No
Quick Patterns → [Match?] → Return Result
    ↓ No
Rule-Based → [Match?] → Return Result
    ↓ No
LLM Classification → Return Result
```

## 📈 Performance

Typical performance on a modern machine:

- **Cached queries**: ~1ms
- **Pattern matches**: ~1-5ms
- **Rule matches**: ~5-10ms
- **LLM calls**: ~1000-3000ms (first time)
- **Subsequent LLM calls**: ~500-1500ms (cached prompts)

## 🔍 Troubleshooting

### Common Issues

1. **"Connection refused" error**
   - Ensure Ollama is running: `ollama serve`
   - Check if the endpoint is correct

2. **Slow performance**
   - Use a faster model like `tinyllama:latest`
   - Increase cache size
   - Add more rule-based patterns

3. **Incorrect classifications**
   - Update `config.yaml` with domain-specific rules
   - Add more examples to the config
   - Use a more accurate model

4. **JSON parsing errors**
   - The LLM might be returning malformed JSON
   - Try a different model
   - Check the system prompt

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🧪 Testing

Run the example script:

```bash
python example_usage.py
```

This will test various query types and show performance metrics.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request



## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM inference
- Inspired by RAG system query filtering needs

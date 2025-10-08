# Elite RAG Orchestrator - Quick Reference Card

## 🚀 Quick Start

```bash
# Initialize and run
python query.py

# Demo mode
python query.py --demo
```

---

## 📊 Core Concepts

### Hybrid Search Formula
```
final_score = α × dense_score + (1-α) × bm25_score
```

- `α = 0.0` → Pure BM25 (keyword)
- `α = 0.5` → Balanced (default)
- `α = 1.0` → Pure dense (semantic)

### Intent Types

| Intent | Triggers | Example |
|--------|----------|---------|
| **Definition** | what is, define, explain | "What is PPU?" |
| **Lookup** | numbers, specs | "PPU voltage specification?" |
| **Troubleshooting** | error, fix, issue | "Fix print quality artifacts" |
| **Reasoning** | how to, steps | "How to install printhead?" |
| **Comparison** | vs, compare, difference | "Compare inline vs standard degasser" |

---

## 💻 Python API

### Basic Usage
```python
from query import EliteRAGQuery

rag = EliteRAGQuery()
rag.initialize()

response = rag.query("How to troubleshoot nozzle clogging?")
print(rag.format_response(response))
```

### Advanced Options
```python
response = rag.query(
    query="PPU installation steps",
    top_k=15,              # Retrieve 15 chunks
    alpha=0.3,             # Favor BM25 (keyword)
    dynamic_windowing=True # Adaptive chunk selection
)
```

### Metadata Filtering
```python
# Query only tables
response = rag.query(
    query="temperature specifications",
    metadata_filters={"content_type": "table"}
)

# Query specific document
response = rag.query(
    query="troubleshooting",
    metadata_filters={"file_name": "DuraFlex Troubleshooting Guide.pdf"}
)
```

---

## 🔧 CLI Advanced Options

```bash
# Adjust hybrid search weight
alpha:0.3 how to install printhead?

# Retrieve more chunks
top:20 troubleshooting guide

# Combine options
alpha:0.3 top:15 compare degasser types
```

---

## 📈 Alpha Parameter Guide

| Use Case | α Value | Why |
|----------|---------|-----|
| Part numbers, codes | 0.2-0.3 | Exact match critical |
| Technical specs | 0.3-0.4 | Terms + some semantic |
| Troubleshooting | 0.4-0.6 | Balanced approach |
| Conceptual questions | 0.7-0.8 | Semantic understanding |
| General queries | 0.5 | Default balanced |

---

## 📋 Response Structure

```
================================================================================
ANSWER:
================================================================================
According to [Source] [1]:
[Grounded content with citations]

================================================================================
REASONING SUMMARY:
================================================================================
Retrieved N chunks. Intent: [type] (confidence: XX%).
Average relevance: 0.XXX

================================================================================
SOURCE SUMMARY:
================================================================================
[1] Document.pdf (pages: 1, 3, 5) [table]
[2] Guide.pdf (pages: 12)
```

---

## 🎯 Components

### RAGOrchestrator
Main orchestration engine

**Key Method:**
```python
orchestrate_query(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    metadata_filters: Optional[Dict] = None,
    dynamic_windowing: bool = True
) -> StructuredResponse
```

### IntentClassifier
Classifies query intent

```python
from orchestrator import IntentClassifier

classifier = IntentClassifier()
intent = classifier.classify("How to fix error?")
# QueryIntent(intent_type='troubleshooting', confidence=0.85)
```

### QueryRewriter
Enhances queries

```python
from orchestrator import QueryRewriter

rewriter = QueryRewriter()
cleaned = rewriter.clean_query("printeer temprature")
# "printer temperature"

expanded = rewriter.expand_acronyms("PPU setup")
# "PPU (printhead power unit) setup"
```

### HybridRetriever
Combines dense + BM25

```python
from orchestrator import HybridRetriever

retriever = HybridRetriever(index, embed_model, reranker)
nodes = retriever.hybrid_search(
    query="installation",
    top_k=10,
    alpha=0.5
)
```

---

## ⚡ Performance

- **Index Load**: 5-10s (one-time)
- **BM25 Init**: 2-5s (one-time)
- **Query**: 500-2000ms
  - Dense: ~200ms
  - BM25: ~50ms
  - Re-rank: ~300ms
  - Generate: ~100ms

---

## 🔍 Troubleshooting

### Low relevance scores
```python
# Try more chunks
response = rag.query(query, top_k=20)

# Adjust alpha
response = rag.query(query, alpha=0.3)  # or 0.7
```

### Missing expected sources
```bash
# Rebuild index
python ingest.py
```

### Slow processing
```python
# Reduce chunks
response = rag.query(query, top_k=5)

# Or disable re-ranking
orchestrator.reranker = None
```

---

## 📂 File Structure

```
.
├── orchestrator.py          # Core RAG orchestration
├── query.py                 # CLI interface
├── ingest.py                # Document ingestion
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── RAG_ORCHESTRATOR_GUIDE.md  # Full documentation
└── QUICK_REFERENCE.md       # This file
```

---

## 🎓 Examples

### Example 1: Troubleshooting
```python
response = rag.query(
    "How to fix print quality artifacts?",
    alpha=0.5  # Balanced
)
```

### Example 2: Technical Spec
```python
response = rag.query(
    "PPU voltage specification",
    alpha=0.3,  # Favor exact terms
    metadata_filters={"content_type": "table"}
)
```

### Example 3: Procedure
```python
response = rag.query(
    "Steps to install inline degasser",
    alpha=0.6,  # Slight semantic preference
    top_k=15    # More context
)
```

### Example 4: Comparison
```python
response = rag.query(
    "Compare inline degasser vs standard degasser",
    alpha=0.7,  # Semantic understanding
    dynamic_windowing=True
)
```

---

## 🛠️ Customization

### Add Custom Acronym
```python
# In orchestrator.py → QueryRewriter.__init__()
self.acronym_map['xyz'] = 'your expansion'
```

### Add Custom Intent
```python
# In orchestrator.py → IntentClassifier.classify()
elif 'install' in query_lower:
    intent_type = 'installation'
    confidence = 0.85
```

### Adjust Dynamic Windowing
```python
# In orchestrator.py → _apply_dynamic_windowing()
threshold = mean_score - 0.5 * std_score  # Change 0.5
```

---

## 📊 Confidence Interpretation

- **>80%**: High confidence, trust answer
- **60-80%**: Medium, verify with sources
- **<60%**: Low, insufficient data

---

## 🔗 Key Models

- **Embeddings**: `BAAI/bge-large-en-v1.5`
- **Re-ranker**: `BAAI/bge-reranker-large`
- **BM25**: Okapi BM25 algorithm

---

## 📚 Full Documentation

See `RAG_ORCHESTRATOR_GUIDE.md` for complete documentation.

---

**Version:** 1.0.0  
**Updated:** October 2025


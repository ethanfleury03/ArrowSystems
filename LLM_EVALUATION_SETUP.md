# LLM Document Evaluation Setup Guide

This guide explains how to set up and use the new LLM-based document evaluation feature in your RAG system.

## 🚀 Quick Start

### 1. Install Ollama

**Windows:**
```bash
# Download from https://ollama.ai/
# Or use winget:
winget install Ollama.Ollama
```

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Download a Model

```bash
# Download Llama 3.1 8B (recommended for document evaluation)
ollama pull llama3.1:8b

# Alternative models:
ollama pull mistral:7b
ollama pull codellama:7b
```

### 3. Start Ollama Service

```bash
# Start Ollama (runs in background)
ollama serve
```

### 4. Test the Feature

```bash
# Run the test script
python test_llm_evaluation.py
```

## 🔧 Configuration

The LLM evaluation feature is configured in `config.yaml`:

```yaml
# LLM Document Evaluation Configuration
llm_evaluation:
  enabled: true                    # Enable/disable LLM evaluation
  model: "llama3.1:8b"            # Ollama model name
  max_documents: 10               # Max documents to evaluate per query
  confidence_threshold: 0.6       # Min confidence to use LLM evaluation
  enable_caching: true            # Cache evaluation results
  temperature: 0.1               # Low temperature for consistency
  max_tokens: 500                # Max tokens for LLM response
```

## 🎯 How It Works

### 1. **Anti-Hallucination Measures**
- **Constrained Prompting**: Forces LLM to only use information from documents
- **Fact Validation**: Cross-checks claimed facts against source documents
- **Confidence Scoring**: Only uses high-confidence evaluations (>0.6)
- **Conservative Scoring**: Encourages lower scores when uncertain

### 2. **Evaluation Process**
1. **Document Retrieval**: Standard hybrid search (dense + BM25)
2. **LLM Evaluation**: Each document is evaluated for relevance
3. **Score Adjustment**: Original scores combined with LLM scores (70% original, 30% LLM)
4. **Re-ranking**: Documents re-ranked based on new scores
5. **Caching**: Results cached to avoid repeated evaluations

### 3. **Prompt Template**
```
TASK: Evaluate document relevance to query with ZERO hallucinations.

CONSTRAINTS:
- Only use information explicitly present in the document
- Do not add external knowledge or assumptions
- Score must be between 0.0 and 1.0
- Be conservative with scoring
- If uncertain, use lower scores

EVALUATION CRITERIA:
1. Direct relevance to query (0.0-0.4)
2. Completeness of information (0.0-0.3)
3. Clarity and specificity (0.0-0.3)
```

## 📊 Performance Impact

### **Benefits:**
- ✅ **Better Relevance**: LLM understands context and intent
- ✅ **Reduced Hallucinations**: Constrained prompting prevents false information
- ✅ **Caching**: Repeated evaluations are cached for speed
- ✅ **Fallback**: Falls back to standard ranking if LLM fails

### **Costs:**
- ⚠️ **Latency**: Adds 2-5 seconds per query (with caching)
- ⚠️ **Resource Usage**: Requires local GPU/CPU for Ollama
- ⚠️ **Memory**: Model needs 4-8GB RAM for 8B models

## 🛠️ Usage Examples

### **Enable in Code:**
```python
from orchestrator import RAGOrchestrator

# Initialize with LLM evaluation enabled
orchestrator = RAGOrchestrator(enable_llm_evaluation=True)

# Standard query (now with LLM evaluation)
response = orchestrator.orchestrate_query("How to troubleshoot temperature issues?")
```

### **Disable LLM Evaluation:**
```python
# Initialize without LLM evaluation
orchestrator = RAGOrchestrator(enable_llm_evaluation=False)

# Or disable in config.yaml
llm_evaluation:
  enabled: false
```

## 🔍 Monitoring and Debugging

### **Log Messages:**
```
🤖 Applying LLM document evaluation to 8 documents
✅ Document evaluated: score=0.847, confidence=0.923
⚠️ Low confidence evaluation (0.456), using original score
```

### **Cache Statistics:**
```python
# Get cache stats
stats = orchestrator.document_evaluator.get_cache_stats()
print(f"Cached evaluations: {stats['cached_evaluations']}")

# Clear cache
orchestrator.document_evaluator.clear_cache()
```

## 🚨 Troubleshooting

### **Common Issues:**

1. **"Ollama not available"**
   - Install Ollama: https://ollama.ai/
   - Start service: `ollama serve`
   - Check if model is downloaded: `ollama list`

2. **"Low confidence evaluations"**
   - Try a different model: `ollama pull mistral:7b`
   - Adjust confidence threshold in config
   - Check if documents are relevant to query

3. **"Evaluation failed"**
   - Check Ollama is running: `ollama ps`
   - Verify model is available: `ollama list`
   - Check system resources (RAM/GPU)

4. **"Slow performance"**
   - Enable caching in config
   - Reduce max_documents
   - Use smaller model (7B instead of 13B)

## 📈 Advanced Configuration

### **Custom Models:**
```yaml
llm_evaluation:
  model: "mistral:7b"  # Try different models
  temperature: 0.05    # Even lower for more consistency
  max_tokens: 300      # Shorter responses
```

### **Performance Tuning:**
```yaml
llm_evaluation:
  max_documents: 5     # Evaluate fewer documents
  confidence_threshold: 0.8  # Only use very confident evaluations
  enable_caching: true # Always enable caching
```

## 🎉 Success Indicators

You'll know the feature is working when you see:
- ✅ "Ollama client initialized" in logs
- ✅ "Applying LLM document evaluation" messages
- ✅ Higher relevance scores for good matches
- ✅ Lower scores for irrelevant documents
- ✅ Cached evaluations in subsequent queries

## 📚 Next Steps

1. **Test with your documents**: Run queries on your actual knowledge base
2. **Monitor performance**: Check if evaluation improves results
3. **Tune parameters**: Adjust confidence thresholds and model settings
4. **Scale up**: Consider using larger models for better accuracy

---

**Need help?** Check the logs for detailed error messages and ensure Ollama is properly installed and running.

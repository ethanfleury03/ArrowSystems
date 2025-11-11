# Claude Semantic Rewriting for RAG Ingestion

## Overview

An optional Claude-powered semantic rewriting step has been added to the RAG ingestion pipeline. This feature uses Claude API to improve text clarity and remove redundancies while preserving all technical meaning and structured content.

## Features

### ✅ What It Does

- **Semantically rewrites text chunks** to improve clarity and remove filler phrases
- **Preserves technical accuracy** - all technical terms, specifications, and measurements remain unchanged
- **Preserves structured content** - tables, code blocks, and lists are kept as-is
- **Maintains metadata** - all chunk metadata (file_name, page_label, chunk_index, etc.) is preserved
- **Graceful fallback** - if Claude fails for a chunk, the original chunk is used
- **Comprehensive logging** - tracks which chunks were rewritten, skipped, or failed

### 🚫 What It Doesn't Do

- Does NOT rewrite tables, code blocks, or dense lists (preserved as-is)
- Does NOT modify chunk boundaries or sizes
- Does NOT change metadata structure
- Does NOT rewrite chunks shorter than 100 characters (skipped for efficiency)

## Configuration

Add to your `config.yaml`:

```yaml
# Claude Semantic Rewriting Configuration (Optional)
claude_rewriting:
  enabled: false              # Set to true to enable Claude rewriting
  model: "claude-3-5-sonnet-20241022"  # Claude model to use
  api_key: null              # Leave null to use ANTHROPIC_API_KEY env var
  max_retries: 2             # Retry attempts per chunk
  timeout: 30                # Request timeout in seconds
```

### Environment Variable

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Or in your `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

## Usage

### Enable Claude Rewriting

1. **Set API key** (if not using env var):
   ```yaml
   claude_rewriting:
     enabled: true
     api_key: "your-api-key"
   ```

2. **Or use environment variable**:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key"
   ```
   Then set `enabled: true` in config.

3. **Run ingestion**:
   ```bash
   python -m backend.ingest
   ```

### Disable Claude Rewriting

Set `enabled: false` in config (default) or remove the `claude_rewriting` section entirely.

## Processing Pipeline

The rewriting step happens **after preprocessing and chunking**, **before embedding**:

```
[Step 1/7] Load PDFs
[Step 2/7] Enhanced preprocessing (TOC removal, artifact fixing)
[Step 3/7] Extract non-text content (tables, images)
[Step 4/7] Create non-text nodes
[Step 5/7] Smart chunking and filtering
[Step 5.5/7] 🤖 Claude semantic rewriting (OPTIONAL - only if enabled)
[Step 6/7] Generate embeddings
[Step 7/7] Save index
```

## What Gets Rewritten

### ✅ Rewritten (Regular Text)
- Technical explanations
- Procedural instructions
- Descriptive paragraphs
- Mixed content with some structure

### 🚫 Preserved As-Is (Structured Content)
- **Tables** - Markdown tables, tab-separated data
- **Code blocks** - Code snippets, command examples
- **Dense lists** - Lists where 50%+ lines are list items
- **Images/Captions** - Image metadata nodes
- **Short chunks** - Chunks < 100 characters (not worth API call)

## Example Output

### Before Rewriting:
```
The printhead assembly should be installed carefully. It is important to note that 
the printhead must be level within 0.5mm. Please note that improper installation 
may cause damage. As you can see from the diagram, the alignment is critical.
```

### After Rewriting:
```
Install the printhead assembly carefully. The printhead must be level within 0.5mm. 
Improper installation may cause damage. As shown in the diagram, alignment is critical.
```

**Changes:**
- Removed redundant phrases ("It is important to note that", "Please note that")
- Improved sentence flow
- Preserved technical specifications (0.5mm)
- Maintained meaning

## Logging & Statistics

During ingestion, you'll see:

```
[Step 5.5/7] 🤖 Claude semantic rewriting (improving clarity while preserving meaning)...
   - This step uses Claude API to enhance text clarity
   - Structured content (tables, code, lists) will be preserved as-is
   - Estimated time: 1-3 minutes per 100 chunks
   
   ✅ Rewriting complete:
      - Rewritten: 142 chunks
      - Preserved (structured): 23 chunks
      - Skipped: 5 chunks
      - Failed (using original): 2 chunks
```

## Performance

### Time Estimate
- **~1-3 minutes per 100 chunks** (depends on chunk length and API latency)
- For ~200 chunks: ~2-6 minutes additional time

### Cost Estimate (Claude Sonnet)
- **Input:** ~$3 per 1M tokens
- **Output:** ~$15 per 1M tokens
- **Typical chunk:** ~500-1000 tokens input, ~400-800 tokens output
- **Cost per chunk:** ~$0.001-0.002
- **For 200 chunks:** ~$0.20-0.40

### API Rate Limits
- Claude API has rate limits (check Anthropic dashboard)
- The rewriter includes retry logic (2 attempts by default)
- Failed chunks fall back to original (no data loss)

## Error Handling

### Automatic Fallbacks

1. **API Failure** → Uses original chunk
2. **Timeout** → Retries up to `max_retries` times, then uses original
3. **Invalid Response** → Validates response length, uses original if too short
4. **Missing API Key** → Disables rewriting automatically with warning

### Logging

All failures are logged:
- `logger.warning()` for retries and fallbacks
- `logger.debug()` for skipped structured content
- Statistics tracked in final summary

## Integration Points

### Compatible With

✅ Existing preprocessing (TOC removal, artifact fixing)  
✅ Smart chunking (1536 chars, 256 overlap)  
✅ Metadata preservation (all fields maintained)  
✅ Vector store format (JSON output unchanged)  
✅ Embedding model (BAAI/bge-large-en-v1.5)  

### No Changes To

❌ Chunk sizes or boundaries  
❌ Metadata schema  
❌ Vector store structure  
❌ Embedding dimensions  
❌ Retrieval logic  

## Best Practices

### When to Enable

✅ **Enable** if:
- You want maximum retrieval quality
- Your documents have verbose/redundant language
- You have Claude API budget
- You're rebuilding the index anyway

❌ **Disable** if:
- Documents are already concise
- API costs are a concern
- You're doing frequent re-indexing
- Documents are mostly structured (tables/code)

### Cost Optimization

1. **Test on small subset first** - Enable for 10-20 chunks to estimate cost
2. **Use Sonnet** - Cheaper than Opus, quality is excellent
3. **Skip short chunks** - Already implemented (< 100 chars)
4. **Preserve structured content** - Already implemented (tables/code)

## Troubleshooting

### "Claude rewriting enabled but ANTHROPIC_API_KEY not found"

**Solution:** Set the API key:
```bash
export ANTHROPIC_API_KEY="your-key"
```

### "Failed to initialize Claude client"

**Solution:** Check:
1. API key is valid
2. Anthropic package is installed: `pip install anthropic`
3. Network connectivity

### "Claude rewrite failed after 2 attempts"

**Solution:** 
- Check API rate limits
- Verify API key permissions
- Check network connectivity
- Original chunk will be used (no data loss)

### High failure rate

**Possible causes:**
- API rate limiting
- Network issues
- Invalid API key
- Chunks too long (exceeds token limits)

**Solution:** Check logs for specific error messages.

## Code Structure

### Key Classes

- **`ClaudeSemanticRewriter`** - Main rewriting class
  - `rewrite_chunk()` - Rewrite single chunk
  - `rewrite_nodes()` - Batch rewrite with progress tracking
  - `_is_structured_content()` - Detect tables/code/lists
  - `_create_rewrite_prompt()` - Generate Claude prompt

### Integration

- Initialized in `TechnicalRAGPipeline.__init__()`
- Called in `build_index()` between filtering and embedding
- Configurable via `config.yaml`

## Future Enhancements

Potential improvements:
- Batch API calls for better throughput
- Caching rewritten chunks (avoid re-rewriting unchanged content)
- Custom prompts per content type
- Parallel processing with rate limit handling
- Cost tracking and reporting


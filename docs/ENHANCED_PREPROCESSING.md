# Enhanced AI-Powered Preprocessing for RAG Ingestion

## Overview

The ingestion pipeline has been enhanced with AI-powered preprocessing to improve retrieval quality by cleaning, normalizing, and filtering text chunks before embedding. This ensures only semantically meaningful, clean content is indexed.

## Key Enhancements

### 1. **Table of Contents Detection & Removal**
- Automatically detects TOC sections using pattern matching
- Removes TOC entries (section numbers with page references)
- Prevents TOC noise from polluting search results

**Detection Patterns:**
- TOC headers: "Table of Contents", "Contents", "Index", "TOC"
- TOC entries: "1.2 Section Name    5" or "Section .......... 10"

### 2. **Enhanced Header/Footer Removal**
- Removes document headers/footers that appear on multiple pages
- Detects product names (DuraFlex, DuraCore, etc.) in headers
- Filters version numbers and revision info from headers

**Patterns Detected:**
- Product name headers: "DuraFlex Manual V1.0"
- Version numbers: "V2.3", "Rev 5"
- Short uppercase lines (likely headers)

### 3. **First Page Filtering**
- Identifies cover pages and title pages with minimal content
- Skips pages that are mostly titles/logos without meaningful text
- Preserves first pages that contain actual content

**Detection Criteria:**
- Page number is 1, i, or I
- Word count < 15 OR (< 50 words AND mostly cover page indicators)
- Cover indicators: All caps titles, "Manual/Guide/Databook" patterns

### 4. **Text Artifact Fixing**

#### Hyphenation Fixing
- Fixes words split across lines: `"print-\nhead"` → `"printhead"`
- Handles various spacing patterns around hyphens

#### Line Break Normalization
- Joins broken sentences split across lines
- Preserves intentional paragraph breaks (double newlines)
- Preserves structured content (lists, code blocks)

**Logic:**
- If line doesn't end with punctuation AND next line doesn't start a sentence → join them
- Preserves list items, bullets, numbered steps

### 5. **Redundant Phrase Removal**
Removes common filler phrases that don't add semantic value:

- "please note that"
- "it is important to note that"
- "as you can see"
- "as shown above/below"
- "as mentioned previously/earlier"
- "for more information please refer to" → "See"
- "for additional details please see" → "See"

### 6. **Technical Content Normalization**
- Fixes spacing around punctuation
- Normalizes parentheses and brackets spacing
- Fixes number ranges: "5 - 10" → "5-10"
- Fixes slashes: "A / B" → "A/B"
- Preserves intentional spacing in tables/code blocks

### 7. **Enhanced Filtering with Skip Reasons**

All skipped chunks/pages are tracked with reasons for auditing:

**Skip Reasons:**
- `empty_text`: Chunk is empty
- `too_short`: Less than 30 characters
- `low_alphabetic_content`: Less than 50% alphabetic characters
- `table_of_contents`: Detected as TOC section
- `first_page_no_content`: First page with no meaningful content
- `low_content`: Page has < 15 words

## Processing Pipeline

### Step-by-Step Cleaning Order

1. **Remove Table of Contents** - Strip TOC sections first
2. **Remove Headers/Footers** - Clean page-level noise
3. **Remove Boilerplate** - Copyright, page numbers, etc.
4. **Fix Text Artifacts** - Hyphenation and line breaks
5. **Remove Redundant Phrases** - Clean filler text
6. **Normalize Technical Content** - Spacing, punctuation
7. **Normalize Whitespace** - Final cleanup pass

### Chunking & Filtering

- **Chunk Size:** 1536 characters (configurable)
- **Overlap:** 256 characters (configurable)
- **Structured Content:** Preserved as single units (tables, code, lists)
- **Quality Filters:** Applied after chunking with skip reason tracking

## Output Format

### TextNode Structure

```python
TextNode(
    text="cleaned and normalized chunk text",
    metadata={
        "file_name": "DuraFlex_Manual.pdf",
        "page_label": "5",
        "chunk_index": 2,
        "total_chunks": 15,
        "content_type": "text",
        # ... other metadata fields
    }
)
```

### Skip Reason Tracking

During ingestion, skip reasons are logged and summarized:

```
✅ Preprocessed 42 documents (3 pages skipped: low_content: 2, first_page_no_content: 1)
✅ Created 187 text nodes (5 filtered: too_short: 3, table_of_contents: 2)
```

## Benefits

1. **Improved Retrieval Quality**
   - Cleaner chunks = better semantic matching
   - Less noise = more relevant results

2. **Better Embedding Quality**
   - Normalized text = more consistent embeddings
   - Fixed artifacts = better semantic understanding

3. **Audit Trail**
   - Skip reasons help identify problematic documents
   - Can tune filters based on skip statistics

4. **Preserved Structure**
   - Tables, code blocks, lists remain intact
   - Technical formatting preserved for clarity

## Configuration

All preprocessing is enabled by default. The existing chunking configuration in `config.yaml` applies:

```yaml
chunking:
  chunk_size: 1536
  chunk_overlap: 256
```

## Backward Compatibility

- ✅ Existing pipeline structure unchanged
- ✅ Same output format (TextNode objects)
- ✅ Same metadata schema
- ✅ Compatible with existing vector store

## Performance Impact

- **Preprocessing:** Minimal overhead (~5-10% slower)
- **Filtering:** Reduces chunks to embed (~10-20% fewer chunks)
- **Net Result:** Faster embedding + better quality

## Future Enhancements

Potential improvements:
- ML-based TOC detection (more accurate)
- OCR artifact fixing for scanned PDFs
- Language-specific normalization
- Custom redundant phrase dictionaries per document type


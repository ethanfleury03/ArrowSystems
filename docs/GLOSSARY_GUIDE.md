# Glossary Feature Guide

## Overview

The glossary feature provides specialized handling for technical terminology, acronyms, and definitions. Instead of treating glossary entries like regular documents, they receive special treatment for:

- **Query augmentation** - Expand queries with aliases and synonyms
- **Fast definitional answers** - Answer "what is X?" queries instantly from glossary
- **Acronym expansion** - Auto-populate the acronym map
- **Better recall** - Improve semantic + keyword matching for domain terminology

## Benefits

### Why Use a Separate Glossary?

For ~100 terms, keeping a dedicated glossary index provides:

1. **Precision**: One entry per term → crisp, authoritative definitions
2. **Control**: Route definitional queries to glossary first
3. **Query boost**: Use aliases/synonyms to expand queries before hybrid retrieval
4. **Fast**: 100 entries is tiny and instant to search
5. **Maintainable**: Update terminology in one place

## Setup

### 1. Create Your Glossary File

**Recommended: CSV Format**

Create `data/glossary.csv` with columns: `term`, `definition`, `aliases`

```csv
term,definition,aliases
PPU,Printhead Power Unit - provides electrical power to the printhead assembly,ppu|printhead power unit
DPI,Dots Per Inch - measurement of print resolution,dpi|dots per inch
CLI,Command Line Interface - text-based user interface,cli|command line|cmd
Firmware,Software embedded in hardware devices that controls basic operations,fw|firm ware
Substrate,Material being printed on such as fabric or vinyl,substrate|media|material
RIP,Raster Image Processor - converts vector graphics to raster format,rip|raster processor
```

**Format Rules:**
- **Aliases**: Pipe-separated (e.g., `ppu|printhead power unit`)
- **No commas in text**: Use semicolons or periods instead
- **UTF-8 encoding**: For special characters

**Alternative: PDF Format**

If you have a PDF glossary, format it with simple lines:

```
Term: Definition goes here
Another Term: Another definition here
```

The loader will parse "Term: Definition" or "Term - Definition" patterns.

### 2. Place the File

Put your glossary file in the `data/` directory:

```
data/
  ├── glossary.csv          ← Your glossary here
  ├── DuraFlex_Manual.pdf
  └── Troubleshooting.pdf
```

### 3. Configure in `config.yaml`

Enable and configure the glossary:

```yaml
glossary:
  enabled: true              # Enable glossary features
  path: "data/glossary.csv"  # Path to your glossary file
  auto_populate_acronyms: true
```

**Configuration Options:**

- `enabled` (bool): Turn glossary on/off without deleting the file
- `path` (string): Relative or absolute path to glossary CSV/PDF
- `auto_populate_acronyms` (bool): Auto-enrich the acronym map from aliases

### 4. Restart the Application

The glossary index loads when the orchestrator initializes:

```bash
# If using start script
./start.sh

# Or restart your Streamlit app
streamlit run app.py
```

You should see in the logs:

```
✅ Loaded glossary index with 100 entries from data/glossary.csv
```

## How It Works

### Query Augmentation

When you search for "PPU troubleshooting", the system:

1. Searches the glossary for "PPU"
2. Finds aliases: `ppu`, `printhead power unit`
3. Augments query: "PPU troubleshooting (ppu | printhead power unit)"
4. Runs hybrid retrieval with augmented query → better recall

### Definitional Query Fast-Path

Queries like:
- "What is PPU?"
- "Define substrate"
- "What does RIP stand for?"
- "Meaning of CLI"

Are automatically:
1. Detected as `intent_type: definition`
2. Answered **directly from glossary** (no hybrid retrieval)
3. Return instant, crisp definitions

Example response:
```
PPU: Printhead Power Unit - provides electrical power to the printhead assembly

[Answered from glossary]
```

### Acronym Map Auto-Population

Your existing `QueryRewriter` has an `acronym_map` for expanding acronyms. The glossary feature **automatically populates this map** from your glossary aliases, so you don't maintain the same information in two places.

Before:
```python
self.acronym_map = {
    'ppu': 'printhead power unit',
    'cli': 'command line interface',
    # ... manually maintained
}
```

After (with glossary):
```python
# Auto-populated from glossary.csv aliases column
# No manual maintenance needed!
```

### Optional Definition Preface

For non-definitional queries that mention a glossary term, the top definition is prepended to the answer:

**Query**: "How to troubleshoot PPU errors?"

**Answer**:
```
Definition: PPU: Printhead Power Unit - provides electrical power to the printhead assembly

According to Troubleshooting Guide [1]:
If you encounter PPU errors, first check the power connections...
```

## Usage Examples

### Creating Your Glossary

Start with a spreadsheet and export as CSV:

| term | definition | aliases |
|------|-----------|---------|
| PPU | Printhead Power Unit - provides electrical power to the printhead assembly | ppu\|printhead power unit |
| DPI | Dots Per Inch - measurement of print resolution | dpi\|dots per inch |

Save as: `data/glossary.csv`

### Testing Definitional Queries

Once loaded, test with:

```
User: "What is PPU?"
→ Returns glossary definition instantly

User: "Define substrate"
→ Returns glossary definition

User: "What does RIP stand for?"
→ Returns expanded acronym + definition
```

### Testing Query Augmentation

```
User: "PPU troubleshooting steps"
→ Augmented internally to: "PPU troubleshooting steps (ppu | printhead power unit)"
→ Hybrid retrieval finds more relevant chunks
```

## File Format Reference

### CSV Structure

```csv
term,definition,aliases
```

- `term` (required): The canonical term/acronym
- `definition` (required): The definition text
- `aliases` (optional): Pipe-separated alternates

### Sample CSV

```csv
term,definition,aliases
API,Application Programming Interface - allows software to communicate,api|application interface
Bandwidth,Data transfer capacity measured in bits per second,bw|bandwidth|data rate
Cache,Temporary storage for frequently accessed data,cache|cached data
Debug,Process of identifying and fixing software bugs,debug|debugging|troubleshoot
Encryption,Converting data into secure coded format,encrypt|encryption|cipher
```

### PDF Structure

Simple text format with colon or hyphen delimiter:

```
API: Application Programming Interface - allows software to communicate
Bandwidth: Data transfer capacity measured in bits per second
Cache: Temporary storage for frequently accessed data
```

## Troubleshooting

### Glossary Not Loading

**Check the logs** when starting the app:

```
⚠️ Glossary file not found at data/glossary.csv
```

**Solutions**:
- Verify file exists at the configured path
- Check `config.yaml` has correct `path`
- Use absolute path if relative path fails
- Ensure file permissions allow reading

### No Entries Loaded

```
⚠️ No glossary entries loaded
```

**Solutions**:
- Check CSV has proper headers: `term,definition,aliases`
- Ensure rows have both term AND definition filled
- Check for encoding issues (use UTF-8)
- Try opening CSV in a text editor to verify format

### Glossary Not Used in Queries

If glossary seems ignored:

1. Check `config.yaml` has `enabled: true`
2. Restart the application after config changes
3. Check logs for initialization errors
4. Try a definitional query like "What is [term]?"

### Aliases Not Working

If query augmentation isn't happening:

1. Verify aliases column has pipe-separated values: `alias1|alias2`
2. Check logs for glossary load confirmation
3. Test with a query containing an alias term

## Best Practices

### 1. Keep Definitions Concise

✅ Good:
```
PPU,Printhead Power Unit - provides electrical power to printhead,ppu|printhead power
```

❌ Too verbose:
```
PPU,"The Printhead Power Unit, also known as the PPU, is a critical component that provides electrical power to the printhead assembly. This unit is responsible for...",ppu
```

### 2. Use Meaningful Aliases

Include:
- Acronym variants: `ppu`, `p.p.u.`
- Spelled-out forms: `printhead power unit`
- Common misspellings: `print head power`

### 3. One Entry Per Concept

Don't duplicate. Pick the canonical term:

✅ Good:
```
PPU,Printhead Power Unit - provides power to printhead,ppu|printhead power unit
```

❌ Redundant:
```
PPU,Printhead Power Unit - provides power,ppu
Printhead Power Unit,Provides power to printhead,ppu
```

### 4. Update Regularly

As your domain evolves:
- Add new terminology
- Refine definitions based on user queries
- Add aliases discovered from query logs

### 5. Test After Updates

After editing `glossary.csv`:
1. Restart the app
2. Test a definitional query: "What is [new term]?"
3. Verify it appears in logs: `✅ Loaded glossary index with N entries`

## Integration with Existing Features

### Works With Hybrid Retrieval

Glossary augmentation happens **before** hybrid search, so both dense embeddings and BM25 benefit from expanded queries.

### Works With Intent Classification

The `IntentClassifier` detects definitional queries, which trigger glossary fast-path.

### Works With Query Rewriter

Glossary aliases auto-populate the `QueryRewriter.acronym_map`, keeping everything in sync.

### Works With LLM Answer Generation

If a glossary definition is found, it can preface LLM-generated answers for context.

## Configuration Reference

```yaml
glossary:
  enabled: true                    # Enable/disable glossary features
  path: "data/glossary.csv"        # Path to glossary file (CSV or PDF)
  auto_populate_acronyms: true     # Auto-fill acronym_map from aliases
```

## FAQ

**Q: Can I use multiple glossary files?**
A: Currently one file. Combine multiple CSVs into one master glossary.

**Q: What if my glossary has 1000+ terms?**
A: Works fine, but for 1000+ you might want to optimize with Qdrant collection.

**Q: Can I edit the glossary without restarting?**
A: No, restart required to reload. Future enhancement could add hot-reload.

**Q: Does this work with mock mode?**
A: Yes, glossary loads in both mock and production modes.

**Q: Can I disable glossary temporarily?**
A: Yes, set `enabled: false` in `config.yaml` and restart.

**Q: What about non-English terms?**
A: Full UTF-8 support, works with any language.

---

## Summary

The glossary feature gives you:
- **Fast definitional answers** for "what is X" queries
- **Better retrieval** via query augmentation with aliases
- **Auto-populated acronyms** from your glossary
- **Clean separation** between definitions and documentation

**Next Steps:**
1. Create `data/glossary.csv` with your ~100 terms
2. Set `glossary.enabled: true` in `config.yaml`
3. Restart the app
4. Test with: "What is [your term]?"


# 🎭 Mock Mode Guide - UI Development Without GPU

## Quick Start

### Windows
```bash
start_mock.bat
```

### Linux/Mac
```bash
chmod +x start_mock.sh
./start_mock.sh
```

Then open http://localhost:8501 and login with `admin` / `admin123`

---

## What is Mock Mode?

Mock mode lets you develop and test the UI **without needing**:
- ❌ GPU rental (save $$$)
- ❌ Model downloads (~2.3 GB)
- ❌ Vector index creation (30+ min)
- ❌ Long model loading times (30-60 sec)
- ❌ Slow query processing (15-35 sec per query)

Instead you get:
- ✅ **Instant startup** (~1 second)
- ✅ **Fast responses** (~0.5 seconds per query)
- ✅ **Realistic simulated answers** with proper formatting
- ✅ **All UI features testable**

---

## How It Works

The mock system (`mock_rag.py`) mimics the real RAG system by:

1. **Detecting query intent** (troubleshooting, procedural, definition, comparison, lookup)
2. **Generating contextual answers** based on query keywords
3. **Creating realistic sources** from actual DuraFlex document names
4. **Simulating confidence scores** and reasoning explanations
5. **Matching the exact API** of `EliteRAGQuery`

---

## What You Can Test

### ✅ Fully Testable in Mock Mode:

- **UI Layout & Styling**
  - Header, sidebar, tabs
  - Buttons, forms, inputs
  - Color schemes, fonts, spacing

- **User Interactions**
  - Login/logout flow
  - Query submission
  - Tab navigation
  - Export functionality

- **Display Components**
  - Answer formatting with citations
  - Source display with metadata
  - Confidence meters
  - Intent badges
  - Session statistics

- **Advanced Features**
  - Query history
  - Parameter controls (alpha, top_k)
  - Content type filters
  - User profiles

### ⚠️ Cannot Test (Requires Real System):

- **Actual Answer Quality**
  - Real knowledge retrieval
  - Answer accuracy
  - Relevance ranking

- **Performance Metrics**
  - Real query latency
  - GPU utilization
  - Model accuracy

---

## Switching Between Modes

### Method 1: Use Different Scripts

**Mock Mode:**
```bash
start_mock.bat      # Windows
./start_mock.sh     # Linux/Mac
```

**Real Mode:**
```bash
start.bat          # Windows
./start.sh         # Linux/Mac
```

### Method 2: Environment Variable

**Mock Mode:**
```bash
# Windows (PowerShell)
$env:USE_MOCK_RAG="true"
streamlit run app.py

# Windows (CMD)
set USE_MOCK_RAG=true
streamlit run app.py

# Linux/Mac
export USE_MOCK_RAG=true
streamlit run app.py
```

**Real Mode:**
```bash
# Windows (PowerShell)
$env:USE_MOCK_RAG="false"
streamlit run app.py

# Windows (CMD)
set USE_MOCK_RAG=false
streamlit run app.py

# Linux/Mac
export USE_MOCK_RAG=false
streamlit run app.py
```

---

## Visual Indicator

When mock mode is active, you'll see a warning banner at the top:

```
🎭 MOCK MODE ACTIVE - Using simulated responses for UI development. 
   Set USE_MOCK_RAG=false for real RAG system.
```

---

## Example Mock Responses

### Troubleshooting Query
**Input:** "How to fix print quality issues?"

**Mock Output:**
- ✅ Procedural answer with numbered steps
- ✅ Citations to real DuraFlex documents
- ✅ Confidence score (75-95%)
- ✅ Intent: troubleshooting
- ✅ 2-5 source documents

### Definition Query
**Input:** "What is the PPU?"

**Mock Output:**
- ✅ Definition with key characteristics
- ✅ Operating parameters
- ✅ Maintenance intervals
- ✅ Intent: definition
- ✅ Technical specifications

### Comparison Query
**Input:** "Compare inkjet vs thermal printing"

**Mock Output:**
- ✅ Side-by-side comparison
- ✅ Advantages/disadvantages
- ✅ Use case recommendations
- ✅ Intent: comparison
- ✅ Reference to comparison charts

---

## Development Workflow

### Recommended Workflow:

1. **UI Development Phase** (Use Mock Mode)
   - Design layouts
   - Build components
   - Style interface
   - Test user flows
   - Fast iteration (no waiting!)

2. **Integration Testing** (Switch to Real Mode)
   - Rent GPU instance
   - Run `python ingest.py` to build index
   - Set `USE_MOCK_RAG=false`
   - Test with real knowledge base
   - Verify answer quality

3. **Production Deployment**
   - Deploy to GPU server
   - Use real RAG system
   - Monitor performance

---

## Troubleshooting

### Mock mode not working?

**Check:**
1. Is `mock_rag.py` in the project root?
2. Is `USE_MOCK_RAG=true` set correctly?
3. Check logs for "[MOCK MODE]" messages

### Need to test real system?

**Requirements:**
1. GPU with 4GB+ VRAM (or slow CPU mode)
2. Run `python ingest.py` first
3. Set `USE_MOCK_RAG=false`
4. Have `storage/` directory with index

---

## Cost Savings Example

**Without Mock Mode:**
- Rent GPU for UI development: $0.50/hour × 8 hours = **$4.00/day**
- 5 days of UI work = **$20.00**

**With Mock Mode:**
- Local development: **$0.00/day**
- Rent GPU only for final testing: 2 hours × $0.50 = **$1.00**
- **Total savings: $19.00** (95% reduction!)

---

## Tips for UI Development

1. **Start with Mock Mode**
   - Build all UI components locally
   - Test all user interactions
   - Perfect the design

2. **Test Edge Cases**
   - Try different query types (the mock will adapt)
   - Test with long/short queries
   - Try different parameter combinations

3. **Switch to Real Mode Occasionally**
   - Verify integration still works
   - Check real response formatting
   - Ensure UI handles actual data

4. **Use Version Control**
   - Commit UI changes on `ui/improvements` branch
   - Merge to main when stable
   - Easy rollback if needed

---

## Files Added

- `mock_rag.py` - Mock RAG system implementation
- `start_mock.bat` - Windows startup script for mock mode
- `start_mock.sh` - Linux/Mac startup script for mock mode
- `MOCK_MODE_GUIDE.md` - This guide

## Files Modified

- `app.py` - Added mock mode support with environment variable
- `README.md` - Added mock mode documentation to Quick Start

---

## Questions?

**Q: Will mock mode work offline?**  
A: Yes! No internet required (except for initial Streamlit install).

**Q: Can I customize mock responses?**  
A: Yes! Edit `mock_rag.py` to change response templates.

**Q: Does mock mode save query history?**  
A: Yes! Session state works the same as real mode.

**Q: Can I export mock results?**  
A: Yes! All export features work with mock data.

---

**Happy UI Development! 🎨**

No GPU? No problem! 🚀


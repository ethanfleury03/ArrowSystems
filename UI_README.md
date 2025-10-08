# DuraFlex Technical Assistant - UI Documentation

Professional web interface for the DuraFlex Technical Knowledge System.

## Features

### 🔐 Authentication System
- Secure login with hashed passwords
- Role-based access control (Admin/Technician)
- Session management with timeout
- User profile display

### 🔍 Intelligent Query Interface
- Natural language question input
- Advanced search controls (hybrid search weight, chunk count)
- Content type filters (text, tables, images, captions)
- Example queries for quick start
- Query history tracking

### 💡 Rich Results Display
- **Answer Tab**: Formatted answers with inline citations
  - Confidence meter with visual indicators
  - Intent classification badges
  - Key terms extraction
  
- **Sources Tab**: Detailed source references
  - Document names and page numbers
  - Content type indicators
  - Expandable source cards
  - Visual display of extracted tables
  - Image preview for extracted diagrams
  
- **Reasoning Tab**: AI decision transparency
  - Query analysis details
  - Retrieval strategy explanation
  - Confidence metrics
  
- **Context Tab**: Retrieved document chunks
  - Full context with relevance scores
  - Source attribution

### 📊 Visual Content Display
- **Tables**: Interactive dataframes with CSV export
- **Images**: Full resolution with download option
- **Captions**: Figure and diagram descriptions

### 📥 Export Functionality
- **Excel**: Structured data export with multiple sheets
- **Text**: Plain text reports
- **Copy to Clipboard**: Quick sharing

### 📈 Session Statistics
- Real-time query count
- Session duration tracking
- AI model status
- User information display

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Users
Edit `config/users.yaml` to add technician accounts.

**Default accounts:**
- Admin: `admin` / `admin123`
- Tech1: `tech1` / `tech123`
- Tech2: `tech2` / `tech123`

### 3. Run Application
```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## User Guide

### For Technicians

#### Logging In
1. Open application URL
2. Enter username and password
3. Click "Login"

#### Asking Questions
1. Type your question in the text area
2. (Optional) Adjust search settings in sidebar:
   - **Chunks to Retrieve**: More chunks = more context
   - **Search Mode**: Balance between keyword and semantic search
3. Click "Search Knowledge Base"
4. View results in tabs

#### Understanding Results
- **Green Confidence**: High quality answer (>80%)
- **Yellow Confidence**: Medium quality (50-80%)
- **Red Confidence**: Low quality (<50%)

**Query Types:**
- 🔧 **Troubleshooting**: Problem-solving queries
- 📖 **Definition**: "What is..." questions
- 🧠 **Procedural**: "How to..." questions
- ⚖️ **Comparison**: "Compare X vs Y"
- 🔍 **Lookup**: Specific fact retrieval

#### Viewing Tables and Images
1. Go to "Sources" tab
2. Expand source with table/image icon
3. View interactive table or image
4. Download if needed

#### Exporting Results
1. After viewing results, scroll to export section
2. Choose format (Excel, Text)
3. Click download button
4. File saves to your downloads folder

### For Administrators

#### Adding New Users
1. Generate password hash:
```python
import hashlib
password = "new_password"
salt = "arrow_secure_2024"
hash_val = hashlib.sha256((password + salt).encode()).hexdigest()
print(hash_val)
```

2. Add to `config/users.yaml`:
```yaml
new_username:
  email: user@company.com
  name: Full Name
  password: <generated_hash>
  salt: arrow_secure_2024
  role: technician  # or admin
```

3. Restart application

#### Managing Documents
1. Add PDFs to `data/` directory
2. Run ingestion:
```bash
python ingest.py
```
3. Restart application to load new index

#### Monitoring Usage
- Check `rag_handler.log` for query logs
- Review session statistics in app
- Monitor system resources (CPU, RAM, GPU)

## Configuration

### Application Settings
Edit `config/app_config.yaml`:

```yaml
# RAG parameters
rag:
  default_top_k: 10  # Default chunks to retrieve
  default_alpha: 0.5  # Search balance (0=keyword, 1=semantic)
  
# UI features
ui:
  enable_query_history: true
  enable_export: true
  show_confidence: true
  
  example_queries:
    - "Your custom example query"
```

### Theme Customization
Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"  # Brand color
backgroundColor = "#f5f7fa"
textColor = "#262730"
```

## Architecture

```
app.py (Main Entry Point)
│
├── components/
│   ├── auth.py              # Authentication
│   ├── query_interface.py   # Query UI
│   └── results_display.py   # Results rendering
│
├── utils/
│   ├── session_manager.py   # Session state
│   └── export_utils.py      # Export functions
│
├── config/
│   ├── users.yaml           # User credentials
│   └── app_config.yaml      # App settings
│
└── RAG Backend
    ├── query.py             # Query interface
    ├── orchestrator.py      # RAG orchestration
    └── ingest.py            # Document processing
```

## Performance Tips

### For Best Performance
1. **Use GPU**: Much faster inference
2. **Adjust top_k**: Lower for speed, higher for accuracy
3. **Balance alpha**: 0.5 usually best for hybrid search
4. **Cache results**: Re-use previous queries when possible
5. **Limit content types**: Filter to only needed types

### Resource Requirements
- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, GPU (4GB+ VRAM)
- **Optimal**: 32GB RAM, GPU (8GB+ VRAM)

## Troubleshooting

### "Models not loading"
- Check internet connection
- Clear cache: `rm -rf ~/.cache/huggingface/hub`
- Restart application

### "Login failed"
- Verify username/password
- Check `config/users.yaml` format
- Ensure password hash is correct

### "No results found"
- Check if index is built (`storage/` directory exists)
- Run `python ingest.py` to build index
- Verify documents in `data/` directory

### "Slow queries"
- Reduce top_k value
- Enable GPU if available
- Close other applications
- Check system resources

### "Tables/Images not showing"
- Verify `extracted_content/` directory exists
- Run ingestion with table/image extraction
- Check file permissions

## Security Notes

⚠️ **Important for Production:**
1. Change all default passwords
2. Update cookie secret in `.streamlit/secrets.toml`
3. Use HTTPS (SSL/TLS)
4. Add `.streamlit/secrets.toml` to `.gitignore`
5. Don't commit `config/users.yaml` with real passwords
6. Enable firewall and restrict ports
7. Regular security updates

## Keyboard Shortcuts

- **Ctrl/Cmd + Enter**: Submit query (in text area)
- **Esc**: Close modals/dialogs
- **R**: Refresh page (if app freezes)

## Browser Compatibility

✅ **Fully Supported:**
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

⚠️ **Limited Support:**
- Internet Explorer (not recommended)
- Older mobile browsers

## Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Updating Application
```bash
git pull
pip install -r requirements.txt
streamlit run app.py
```

### Backup Before Updates
```bash
# Backup config and data
tar -czf backup_$(date +%Y%m%d).tar.gz config/ storage/ data/
```

## Support

### Getting Help
- Review documentation
- Check logs: `rag_handler.log`
- Contact IT support
- Email: support@arrowsystems.com

### Reporting Issues
Include:
1. Error message screenshot
2. Query that failed
3. Browser and OS version
4. Relevant log entries

## License

© 2025 Arrow Systems Inc  
All Rights Reserved

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Status**: Production Ready


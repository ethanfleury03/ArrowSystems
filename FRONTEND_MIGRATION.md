# Frontend Migration: Streamlit → Next.js

The frontend has been migrated from Streamlit to Next.js for a modern, production-ready chat interface.

## 🚀 Quick Start

### Development (Recommended)

1. **Start the FastAPI backend:**
   ```bash
   # Terminal 1
   python api.py --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start the Next.js frontend:**
   ```bash
   # Terminal 2
   cd frontend
   npm install
   npm run dev
   ```

3. **Open your browser:**
   - Frontend: http://localhost:3000
   - Backend API docs: http://localhost:8000/docs

### Docker Compose (Full Stack)

Run both backend and frontend together:

```bash
docker-compose up
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

### Docker (Backend Only)

The main `Dockerfile` now runs the FastAPI backend (not Streamlit):

```bash
docker build -t rag-app:local .
docker run -p 8000:8000 rag-app:local
```

## 📁 Project Structure

```
├── frontend/           # Next.js frontend application
│   ├── app/           # Next.js App Router pages
│   ├── components/    # React components
│   ├── lib/           # API client
│   └── types/         # TypeScript types
├── api.py             # FastAPI backend (unchanged)
├── docker-compose.yml # Runs both backend + frontend
└── Dockerfile         # Backend-only container
```

## 🔄 Changes Made

### Updated Files
- `Dockerfile` - Now runs FastAPI backend (port 8000) instead of Streamlit
- `docker-compose.yml` - New file to run both services
- `build-local.ps1` - Updated port references
- `run-local.ps1` - Updated port references

### New Files
- `frontend/` - Complete Next.js application
  - ChatGPT-style UI
  - TypeScript + Tailwind CSS
  - Real-time chat interface
  - Markdown rendering

### Deprecated (but kept for reference)
- `app.py` - Old Streamlit frontend (can be removed later)
- `.streamlit/` - Streamlit config (no longer used)

## 🌐 API Integration

The frontend communicates with the FastAPI backend:

```typescript
POST http://localhost:8000/query
Body: { "query": "your question", "top_k": 10, ... }
Response: { "answer": "...", "sources": [...], ... }
```

See `frontend/lib/api.ts` for implementation.

## ⚙️ Environment Variables

### Frontend
Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Backend
Backend environment variables remain the same (see `api.py`).

## 📝 Development Notes

- The backend API is unchanged - it still runs on port 8000
- Frontend runs on port 3000 (Next.js default)
- CORS is already configured in `api.py` to allow all origins
- For production, update CORS settings in `api.py` to specific origins

## 🐛 Troubleshooting

**Frontend can't connect to backend:**
- Ensure backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` environment variable
- For Docker Compose, use `http://backend:8000` (service name)

**Port conflicts:**
- Backend: 8000
- Frontend: 3000
- Old Streamlit: 8501 (no longer used)

## 🚢 Deployment

### Production Build

**Frontend:**
```bash
cd frontend
npm run build
npm start
```

**Backend:**
```bash
python api.py --host 0.0.0.0 --port 8000
```

### Docker Production

Use `docker-compose.yml` with production environment variables.


# Quick Start Guide

## 🚀 Running the Application

### Option 1: Docker Compose (Easiest - Recommended)

Run both backend and frontend together:

```bash
docker-compose up
```

Then open:
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/docs

### Option 2: Local Development

**Terminal 1 - Backend:**
```bash
python api.py --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open: http://localhost:3000

### Option 3: Docker Backend Only

Build and run just the backend:

```bash
docker build -t rag-app:local .
docker run -p 8000:8000 rag-app:local
```

Access the API at: http://localhost:8000/docs

## 📝 Prerequisites

- **For Docker**: Docker Desktop installed
- **For Local Dev**: 
  - Python 3.11+
  - Node.js 20+
  - npm or yarn

## 🔧 Environment Setup

No special environment variables needed for basic operation. The frontend will automatically connect to the backend.

For production or advanced setups, see `FRONTEND_MIGRATION.md`.

## 🎯 What Changed?

- ✅ New Next.js frontend (ChatGPT-style UI)
- ✅ Replaced Streamlit with modern React/TypeScript
- ✅ Backend unchanged (still FastAPI on port 8000)
- ✅ Docker Compose for easy full-stack deployment

## 📚 More Information

- See `FRONTEND_MIGRATION.md` for detailed migration notes
- See `frontend/README.md` for frontend-specific documentation
- See `api.py` for backend API documentation


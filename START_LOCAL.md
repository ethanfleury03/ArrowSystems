# Quick Start - Local Backend

Since Docker builds are slow due to PyTorch downloads, here's the fastest way to get started:

## 🚀 Start Backend Locally (Recommended)

Your venv has all the dependencies installed. Just run:

**Terminal 1 (Backend):**
```bash
python api.py --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend - you're already doing this):**
Keep `npm run dev` running in the frontend directory.

Then open http://localhost:3000 and start using the app!

## Why Local Instead of Docker?

- ✅ **Instant startup** (no build time)
- ✅ **Your dependencies already installed**
- ✅ **Works immediately**
- ❌ Docker takes 5-10 minutes downloading PyTorch (900MB)

## Want to Use Docker Later?

Once the Docker image builds successfully the first time, it will be cached and startup will be instant. But for development, local is faster.

## Access Points

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs



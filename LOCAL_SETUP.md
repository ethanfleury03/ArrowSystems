# Local Development Setup

Since you're running the frontend manually, here's how to get both working together:

## Quick Setup

**You're already running the frontend on port 3000.** Now you just need the backend on port 8000.

### Option 1: Run Backend with Docker (Fix the timeout issue)

The Docker build is failing due to network timeout downloading PyTorch. Try:

```bash
# Retry the build with more time
docker-compose up --build
```

If it still fails, the network might be slow. Try Option 2.

### Option 2: Run Backend Locally (No Docker)

Since you have a venv with all dependencies installed, you can run the backend directly:

**Terminal 1 - Start Backend:**
```bash
python api.py --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend is already running:**
You're running `npm run dev` manually, keep it running.

### Access Points

- **Frontend**: http://localhost:3000 (already working)
- **Backend API**: http://localhost:8000 (starts with above command)
- **Backend Docs**: http://localhost:8000/docs

## Troubleshooting

### Backend won't start?
- Make sure you have the RAG index: `latest_model/` directory exists
- Check that all Python dependencies are installed in your venv
- See error messages in the terminal

### Frontend can't connect to backend?
- Make sure backend is running on port 8000
- Check browser console for errors
- The frontend's API route will try `http://localhost:8000` automatically

### Port already in use?
- Windows: `netstat -ano | findstr ":8000"` to find what's using it
- Linux/Mac: `lsof -i :8000` to find what's using it

## Next Steps

Once both are running:
1. Open http://localhost:3000
2. Try sending a query
3. Watch the backend terminal for logs



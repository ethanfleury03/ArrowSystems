# Development Setup for Hot Reload

## Quick Start - Frontend Development with Hot Reload

The backend and database run in Docker, but you can run the frontend locally for instant hot-reload:

### Step 1: Start the Backend Service
```powershell
# From project root
docker compose up -d backend
```

### Step 2: Run Frontend Locally
```powershell
cd frontend
npm install  # If not already done
npm run dev
```

The frontend will be available at `http://localhost:3000` with **hot-reload enabled** - changes will appear instantly!

### Step 3: Create `.env.local` file
Create `frontend/.env.local` with:
```env
SESSION_SECRET=your-secret-key-here
BACKEND_URL=http://localhost:8000
```

## Alternative: Docker Development with Volumes

If you prefer everything in Docker, use the development override:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

This will mount your source code as volumes for hot-reload.


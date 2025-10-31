# RAG Assistant Frontend

A modern Next.js frontend for the RAG (Retrieval-Augmented Generation) Assistant, built with TypeScript, Tailwind CSS, and a ChatGPT-inspired interface.

## Features

- 🎨 Clean, ChatGPT-style UI
- 💬 Real-time chat interface
- 📱 Fully responsive design
- 🌙 Dark mode support
- ⚡ Fast and optimized with Next.js 14
- 📝 Markdown rendering for responses

## Tech Stack

- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS**
- **Framer Motion** (animations)
- **Lucide React** (icons)
- **Axios** (API calls)
- **React Markdown** (markdown rendering)

## Development

### Prerequisites

- Node.js 20+ 
- npm or yarn

### Install Dependencies

```bash
npm install
```

### Run Development Server

**Option 1: Run frontend only (backend must be running separately)**

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

Make sure the FastAPI backend is running at `http://localhost:8000`.

**Option 2: Use Docker Compose (recommended)**

```bash
# From project root
docker-compose up
```

This runs both backend and frontend together.

### Environment Variables

For local development, create `frontend/.env.local`:

```env
BACKEND_URL=http://localhost:8000
```

Or use `NEXT_PUBLIC_API_URL` (for backwards compatibility):

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Build for Production

```bash
npm run build
npm start
```

## Docker

### Build and Run Frontend Only

```bash
# Build
docker build -t rag-frontend -f frontend/Dockerfile ./frontend

# Run (backend must be accessible)
docker run -p 3000:3000 -e BACKEND_URL=http://host.docker.internal:8000 rag-frontend
```

### Docker Compose (Recommended)

From the project root:

```bash
docker-compose up
```

This starts both backend (port 8000) and frontend (port 3000).

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Project Structure

```
frontend/
├── app/
│   ├── api/
│   │   └── query/
│   │       └── route.ts    # Next.js API route (proxies to backend)
│   ├── page.tsx            # Main chat page
│   ├── layout.tsx          # Root layout
│   └── globals.css         # Global styles
├── components/
│   ├── Chat.tsx            # Chat container component
│   ├── MessageBubble.tsx   # Individual message component
│   └── InputBar.tsx        # Message input component
├── lib/
│   └── api.ts              # API client (uses Next.js API route)
├── types/
│   └── message.ts          # TypeScript types
└── package.json
```

## API Integration

The frontend uses a Next.js API route (`/app/api/query/route.ts`) to proxy requests to the FastAPI backend. This allows:

- ✅ CORS-free communication
- ✅ Works in Docker (service-to-service)
- ✅ Works in local development
- ✅ Hides backend URL from client

The API route forwards requests to:
```
POST http://localhost:8000/query (or BACKEND_URL/query in Docker)
Body: { "query": "your question", "top_k": 10, ... }
Response: { "answer": "...", "sources": [...], ... }
```

## Architecture

```
Browser → Next.js Frontend (port 3000)
           ↓
      Next.js API Route (/api/query)
           ↓
      FastAPI Backend (port 8000)
```

## Troubleshooting

**Frontend can't connect to backend:**
- Ensure backend is running on port 8000
- Check `BACKEND_URL` environment variable
- In Docker Compose, backend is accessible via service name `backend:8000`

**Port conflicts:**
- Backend: 8000
- Frontend: 3000
- Old Streamlit: 8501 (no longer used)

**Build errors:**
- Make sure Node.js 20+ is installed
- Delete `node_modules` and `package-lock.json`, then run `npm install` again

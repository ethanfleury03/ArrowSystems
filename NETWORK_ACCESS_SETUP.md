# Network Access Setup Guide

This document explains how to access the application from other machines on your network.

## Quick Setup

### 1. Start Backend
```bash
# Backend automatically listens on 0.0.0.0:8000 (all network interfaces)
python -m backend.api --dev --reload
# OR for production:
python -m backend.api --host 0.0.0.0 --port 8000
```

### 2. Start Frontend
```bash
# Development mode (listens on 0.0.0.0:3000)
npm run dev

# Production mode (listens on 0.0.0.0:3000)
npm run build
npm start
```

### 3. Access from Network
- Find your machine's IP address (e.g., `192.168.1.100`)
- From another machine, access: `http://192.168.1.100:3000`
- The frontend will automatically detect the IP and connect to `http://192.168.1.100:8000` for the backend

## How It Works

### Client-Side Detection
- The frontend automatically detects the current hostname from `window.location.hostname`
- If accessing from `localhost` or `127.0.0.1`, it uses `localhost:8000`
- If accessing from a network IP (e.g., `192.168.1.100`), it uses `http://192.168.1.100:8000`

### Server-Side Detection
- Next.js API routes detect the backend URL from the request `Host` header
- Same logic: localhost → localhost:8000, network IP → network IP:8000

## Firewall Configuration

### Windows
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Create inbound rules for ports 3000 and 8000
4. Allow TCP connections on these ports

### Linux
```bash
# Ubuntu/Debian
sudo ufw allow 3000/tcp
sudo ufw allow 8000/tcp

# Or use iptables
sudo iptables -A INPUT -p tcp --dport 3000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 8000 -j ACCEPT
```

## Troubleshooting

### Frontend loads but data doesn't fetch
- **Problem**: Backend not accessible from network
- **Solution**: 
  1. Verify backend is running: `curl http://YOUR_IP:8000/health`
  2. Check firewall allows port 8000
  3. Verify backend listens on `0.0.0.0` not `127.0.0.1`

### Cannot connect to backend
- **Problem**: Backend URL detection failed
- **Solution**: 
  1. Check browser console for errors
  2. Verify the detected backend URL is correct
  3. Manually override in browser console:
     ```javascript
     localStorage.setItem('apiBaseUrlOverride', 'http://YOUR_IP:8000');
     ```

### CORS errors
- **Problem**: Backend CORS not configured correctly
- **Solution**: Backend already allows all origins (`allow_origins=["*"]`), so this shouldn't be an issue

## Environment Variables

### Frontend
- `BACKEND_URL`: Override backend URL (server-side)
- `NEXT_PUBLIC_API_URL`: Override backend URL (client-side)
- These take precedence over auto-detection

### Backend
- `--host 0.0.0.0`: Listen on all interfaces (default)
- `--port 8000`: Port to listen on (default)

## Manual Override

If auto-detection doesn't work, you can manually set the backend URL:

### Browser Console
```javascript
// Set custom backend URL
localStorage.setItem('apiBaseUrlOverride', 'http://192.168.1.100:8000');

// Force localhost
localStorage.setItem('useLocalBackend', 'true');
```

### Environment Variables
Create `frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://192.168.1.100:8000
```

## Testing

1. Start backend and frontend
2. Find your IP: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
3. From another machine, access `http://YOUR_IP:3000`
4. Check browser console for network requests
5. Verify requests go to `http://YOUR_IP:8000`

## Notes

- Both dev and production modes now listen on `0.0.0.0` (all interfaces)
- Auto-detection works for both client-side and server-side API calls
- Manual overrides are available if needed
- Backend CORS is configured to allow all origins


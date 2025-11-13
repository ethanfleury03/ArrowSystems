const DEFAULT_SERVER_BACKEND = 'http://backend:8000';
const DEFAULT_CLIENT_BACKEND = 'http://localhost:8000';

const SERVER_BACKEND =
  process.env.SERVER_BACKEND_URL ??
  process.env.NEXT_PUBLIC_SERVER_BACKEND_URL ??
  DEFAULT_SERVER_BACKEND;

// Get client backend URL - detect from current hostname when accessed from network
// This function is called at runtime, not at module load time, so it can access window
const getClientBackendUrl = (): string => {
  // First check environment variables (these are set at build time)
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // If running in browser, detect the current hostname
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname;
    
    // If accessing from localhost or 127.0.0.1, use localhost:8000
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '') {
      return DEFAULT_CLIENT_BACKEND;
    }
    
    // Otherwise, use the same hostname but port 8000 for backend
    // This allows network access: if frontend is at 192.168.1.100:3000,
    // backend will be at 192.168.1.100:8000
    return `http://${hostname}:8000`;
  }
  
  // Fallback for SSR or when window is not available
  return DEFAULT_CLIENT_BACKEND;
};

const getRuntimeOverride = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  const useLocal = window.localStorage.getItem('useLocalBackend');
  if (useLocal === 'true') {
    return DEFAULT_CLIENT_BACKEND;
  }

  const manualOverride = window.localStorage.getItem('apiBaseUrlOverride');
  if (manualOverride) {
    return manualOverride;
  }

  return null;
};

export const resolveInitialBaseUrl = (): string => {
  if (typeof window === 'undefined') {
    return SERVER_BACKEND;
  }
  return getClientBackendUrl();
};

export const resolveApiBaseUrl = (): string => {
  if (typeof window === 'undefined') {
    return SERVER_BACKEND;
  }
  
  // Check for manual override first
  const override = getRuntimeOverride();
  if (override) {
    return override;
  }
  
  // Otherwise use the auto-detected client backend URL (called at runtime)
  return getClientBackendUrl();
};

export const buildApiUrl = (path: string): string => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${resolveApiBaseUrl()}${normalizedPath}`;
};


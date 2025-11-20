import { isDev, isProd } from '@/lib/env';

const DEFAULT_SERVER_BACKEND = process.env.SERVER_BACKEND_URL || process.env.NEXT_PUBLIC_SERVER_BACKEND_URL || 'http://backend:8080';
const DEFAULT_CLIENT_BACKEND = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

const SERVER_BACKEND =
  process.env.SERVER_BACKEND_URL ??
  process.env.NEXT_PUBLIC_SERVER_BACKEND_URL ??
  DEFAULT_SERVER_BACKEND;

// Get client backend URL - detect from current hostname when accessed from network
// This function is called at runtime, not at module load time, so it can access window
const getClientBackendUrl = (): string => {
  // Production: NEXT_PUBLIC_API_URL must be provided (validated at build time)
  if (isProd) {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL;
    if (!apiUrl) {
      // This should never happen if build-time validation worked, but provide fallback
      console.error('NEXT_PUBLIC_API_URL is required in production but was not set');
      throw new Error('API URL not configured. NEXT_PUBLIC_API_URL must be set in production.');
    }
    return apiUrl;
  }

  // Development: Allow env var override, otherwise auto-detect
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (process.env.REACT_APP_API_URL) {
    return process.env.REACT_APP_API_URL;
  }
  
  // If running in browser, detect the current hostname
  if (typeof window !== 'undefined' && window.location) {
    const hostname = window.location.hostname;
    
    // If accessing from localhost or 127.0.0.1, use default
    if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '') {
      return DEFAULT_CLIENT_BACKEND;
    }
    
    // For production/network access, prefer environment variable
    if (process.env.NEXT_PUBLIC_API_URL) {
      return process.env.NEXT_PUBLIC_API_URL;
    }
    // Fallback: use same hostname with port 8080 (GCP default)
    return `http://${hostname}:8080`;
  }
  
  // Fallback for SSR or when window is not available
  return DEFAULT_CLIENT_BACKEND;
};

const getRuntimeOverride = (): string | null => {
  // Production: Disable localStorage overrides for security
  if (isProd) {
    // Clean up any existing overrides
    if (typeof window !== 'undefined') {
      window.localStorage.removeItem('apiBaseUrlOverride');
      window.localStorage.removeItem('useLocalBackend');
    }
    return null;
  }

  // Development: Allow localStorage overrides
  if (typeof window === 'undefined') {
    return null;
  }

  const useLocal = window.localStorage.getItem('useLocalBackend');
  if (useLocal === 'true') {
    if (isDev) {
      console.warn('[DEV] Using localStorage override: useLocalBackend=true');
    }
    return DEFAULT_CLIENT_BACKEND;
  }

  const manualOverride = window.localStorage.getItem('apiBaseUrlOverride');
  if (manualOverride) {
    if (isDev) {
      console.warn('[DEV] Using localStorage override:', manualOverride);
    }
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


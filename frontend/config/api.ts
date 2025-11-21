import { isDev, isProd } from '@/lib/env';

// With IAM proxy architecture, all requests go through Next.js /api routes
// The /api routes handle IAM authentication to the Cloud Run backend
const API_BASE_URL = '/api';

// For backwards compatibility with existing code
const DEFAULT_SERVER_BACKEND = API_BASE_URL;
const DEFAULT_CLIENT_BACKEND = API_BASE_URL;
const SERVER_BACKEND = API_BASE_URL;

// Get client backend URL - now returns /api (Next.js proxy routes)
const getClientBackendUrl = (): string => {
  // All requests go through Next.js API routes which proxy to backend with IAM auth
  return API_BASE_URL;
};

const getRuntimeOverride = (): string | null => {
  // With IAM proxy, all requests must go through /api - no overrides allowed
  // Clean up any existing overrides
  if (typeof window !== 'undefined') {
    window.localStorage.removeItem('apiBaseUrlOverride');
    window.localStorage.removeItem('useLocalBackend');
  }
  return null;
};

export const resolveInitialBaseUrl = (): string => {
  // All requests go through Next.js API proxy
  return API_BASE_URL;
};

export const resolveApiBaseUrl = (): string => {
  // All requests go through Next.js API proxy
  return API_BASE_URL;
};

export const buildApiUrl = (path: string): string => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE_URL}${normalizedPath}`;
};


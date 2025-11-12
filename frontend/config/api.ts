const DEFAULT_SERVER_BACKEND = 'http://backend:8000';
const DEFAULT_CLIENT_BACKEND = 'http://localhost:8000';

const SERVER_BACKEND =
  process.env.SERVER_BACKEND_URL ??
  process.env.NEXT_PUBLIC_SERVER_BACKEND_URL ??
  DEFAULT_SERVER_BACKEND;

const CLIENT_BACKEND =
  process.env.NEXT_PUBLIC_API_URL ??
  process.env.REACT_APP_API_URL ??
  DEFAULT_CLIENT_BACKEND;

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
  return CLIENT_BACKEND;
};

export const resolveApiBaseUrl = (): string => {
  if (typeof window === 'undefined') {
    return SERVER_BACKEND;
  }
  return getRuntimeOverride() ?? CLIENT_BACKEND;
};

export const buildApiUrl = (path: string): string => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${resolveApiBaseUrl()}${normalizedPath}`;
};


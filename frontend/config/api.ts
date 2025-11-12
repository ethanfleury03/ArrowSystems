const REMOTE_BACKEND =
  process.env.NEXT_PUBLIC_API_URL ??
  process.env.REACT_APP_API_URL ??
  'https://api.example.com';

const LOCAL_BACKEND =
  process.env.NEXT_PUBLIC_LOCAL_BACKEND_URL ??
  process.env.REACT_APP_LOCAL_BACKEND_URL ??
  'http://localhost:8000';

const USE_LOCAL =
  process.env.NEXT_PUBLIC_USE_LOCAL_BACKEND === 'true' ||
  process.env.REACT_APP_USE_LOCAL_BACKEND === 'true';

export const API_BASE_URL = USE_LOCAL ? LOCAL_BACKEND : REMOTE_BACKEND;

const getRuntimeOverride = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }

  const useLocal = window.localStorage.getItem('useLocalBackend');
  if (useLocal === 'true') {
    return LOCAL_BACKEND;
  }

  const manualOverride = window.localStorage.getItem('apiBaseUrlOverride');
  if (manualOverride) {
    return manualOverride;
  }

  return null;
};

export const resolveApiBaseUrl = (): string => getRuntimeOverride() ?? API_BASE_URL;

export const buildApiUrl = (path: string): string => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${resolveApiBaseUrl()}${normalizedPath}`;
};


import { isDev, isProd } from '@/lib/env';

// Ingestion safety flag - controls whether ingestion actions are available in the UI
// This should ONLY be true in dedicated GPU ingestion environments, NOT in production
export const ALLOW_APP_INGESTION =
  process.env.NEXT_PUBLIC_ALLOW_APP_INGESTION === "true";

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

/**
 * Build a URL for viewing/downloading a document.
 * Returns a same-origin URL that serves application/pdf.
 * 
 * @param options - Document identifier options
 * @param options.filename - Filename (required if document_id not provided)
 * @param options.document_id - Document ID (optional, preferred if available)
 * @param options.page - Page number for PDF fragment (optional)
 * @returns URL string (e.g., "/api/documents/filename.pdf#page=1")
 */
export function buildDocumentViewUrl(options: {
  filename?: string;
  document_id?: number | string;
  page?: number;
}): string {
  const { filename, document_id, page } = options;
  
  // Prefer document_id if available, otherwise use filename
  // For now, we use filename since the API route expects filename
  // TODO: If backend adds document_id support, use that instead
  const identifier = filename || (document_id ? String(document_id) : '');
  
  if (!identifier) {
    throw new Error('Either filename or document_id must be provided');
  }
  
  // Encode the filename for URL safety
  const encodedFilename = encodeURIComponent(identifier);
  
  // Build the URL using the Next.js API route (which proxies to backend)
  const baseUrl = `${API_BASE_URL}/documents/${encodedFilename}`;
  
  // Add page fragment if provided (works with browser PDF viewers)
  if (page && page > 0) {
    return `${baseUrl}#page=${page}`;
  }
  
  return baseUrl;
}


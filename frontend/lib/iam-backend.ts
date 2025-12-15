/**
 * IAM-authenticated backend proxy utility
 * 
 * This module provides utilities to make authenticated requests to the Cloud Run backend
 * using Google IAM identity tokens. This allows the frontend (running on Cloud Run) to
 * communicate with the backend (also on Cloud Run) without making the backend publicly accessible.
 * 
 * IMPORTANT: This module should ONLY be used in server-side code (API routes).
 * Never import this in client components as it requires google-auth-library.
 */

import { GoogleAuth } from 'google-auth-library';

// Get backend URL from environment variable (required in production)
// In production, NEXT_PUBLIC_API_URL must be set by Cloud Run deployment
const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || 'http://localhost:8080';

if (!BACKEND_URL) {
  throw new Error('NEXT_PUBLIC_API_URL or BACKEND_URL environment variable must be set');
}

/**
 * Make an authenticated request to the backend using Google IAM identity tokens.
 * 
 * @param path - The API path (e.g., '/auth/login', '/query')
 * @param options - Fetch options (method, body, headers)
 * @returns The response from the backend
 */
export async function iamBackendRequest(
  path: string,
  options: {
    method?: string;
    body?: any;
    headers?: Record<string, string>;
    retries?: number; // Optional: number of retries for 503 errors
    retryDelay?: number; // Optional: delay between retries in ms
  } = {}
): Promise<Response> {
  const { method = 'GET', body, headers = {}, retries = 2, retryDelay = 500 } = options;

  // Retry logic for 503 errors (handles Cloud Run cold starts)
  const maxRetries = retries;
  let lastError: any = null;
  
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
    // Initialize GoogleAuth client
    const auth = new GoogleAuth();
    
    // Get an ID token client for the backend URL
    const client = await auth.getIdTokenClient(BACKEND_URL);

    // Prepare the full URL
    const url = `${BACKEND_URL}${path}`;

    // Prepare request configuration
    const requestConfig: any = {
      url,
      method,
      headers: {
        'Content-Type': 'application/json',
        ...headers,
      },
    };

    // Add body if provided
    if (body !== undefined) {
      if (typeof body === 'string') {
        requestConfig.data = body;
      } else {
        requestConfig.data = body;
      }
    }

    // Make the authenticated request
    const response = await client.request(requestConfig);

    // Check if response indicates an error (google-auth-library doesn't throw on HTTP errors)
    const statusCode = response.status || (response as any).statusCode || 200;
    
    // Check if this is a RAG-specific error (should not retry)
    let isRagError = false;
    let ragErrorCode: string | null = null;
    if (statusCode === 503 && response.data) {
      try {
        const errorData = typeof response.data === 'string' ? JSON.parse(response.data) : response.data;
        const code = errorData?.code || errorData?.detail?.code;
        if (code === 'RAG_NOT_INITIALIZED' || code === 'RAG_WARMING' || code === 'RAG_NOT_CONFIGURED') {
          isRagError = true;
          ragErrorCode = code;
        }
      } catch (e) {
        // If parsing fails, treat as transient error
      }
    }
    
    // If 503 and NOT RAG-specific error and we have retries left, retry
    if (statusCode === 503 && !isRagError && attempt < maxRetries) {
      const delay = retryDelay * Math.pow(2, attempt);
      console.warn(`IAM Backend Request 503 (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms:`, {
        path,
        status: statusCode,
      });
      await new Promise(resolve => setTimeout(resolve, delay));
      continue; // Retry the request
    }
    
    // If RAG-specific error, don't retry - return immediately
    if (isRagError) {
      console.warn('IAM Backend Request: RAG error, skipping retries', { 
        path, 
        status: statusCode, 
        code: ragErrorCode 
      });
    }

    // Convert the response to a standard Response object
    // Preserve all headers from the backend response, especially Set-Cookie for auth
    const responseHeaders: Record<string, string> = {};
    
    // Copy important headers from backend response
    if (response.headers) {
      // Preserve Set-Cookie header for authentication
      if (response.headers['set-cookie']) {
        responseHeaders['set-cookie'] = response.headers['set-cookie'];
      }
      // Preserve other useful headers
      const headersToPreserve = ['content-type', 'cache-control', 'etag'];
      headersToPreserve.forEach(headerName => {
        if (response.headers[headerName]) {
          responseHeaders[headerName] = response.headers[headerName];
        }
      });
    }

    // Handle 204 No Content: return Response without body (204 cannot have a body)
    if (statusCode === 204) {
      return new Response(null, {
        status: 204,
        headers: responseHeaders,
      });
    }

    // For other statuses, include JSON body
    // Only set Content-Type if we're actually returning JSON
    if (!responseHeaders['content-type']) {
      responseHeaders['Content-Type'] = 'application/json';
    }

    return new Response(JSON.stringify(response.data), {
      status: statusCode,
      headers: responseHeaders,
    });
    } catch (error: any) {
      lastError = error;
      
      // Handle network errors and connection failures (common during cold starts)
      // Check if this is a retryable error
      const statusCode = error.response?.status || error.status || (error.code === 'ECONNREFUSED' ? 503 : 500);
      
      // Check if this is a RAG-specific error (should not retry)
      let isRagError = false;
      let ragErrorCode: string | null = null;
      if (statusCode === 503 && error.response?.data) {
        try {
          const errorData = typeof error.response.data === 'string' 
            ? JSON.parse(error.response.data) 
            : error.response.data;
          const code = errorData?.code || errorData?.detail?.code;
          if (code === 'RAG_NOT_INITIALIZED' || code === 'RAG_WARMING' || code === 'RAG_NOT_CONFIGURED') {
            isRagError = true;
            ragErrorCode = code;
          }
        } catch (e) {
          // If parsing fails, treat as transient error
        }
      }
      
      const isRetryable = (statusCode === 503 && !isRagError) || error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT';
      const hasRetriesLeft = attempt < maxRetries;
      
      if (isRetryable && hasRetriesLeft) {
        // Exponential backoff: 500ms, 1000ms, 2000ms, etc.
        const delay = retryDelay * Math.pow(2, attempt);
        console.warn(`IAM Backend Request error (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms:`, {
          path,
          error: error.message,
          code: error.code,
          status: statusCode,
        });
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, delay));
        continue; // Retry the request
      }
      
      // If not a 503, or no retries left, handle the error
      console.error('IAM Backend Request Error:', {
        path,
        error: error.message,
        status: statusCode,
        data: error.response?.data,
        attempt: attempt + 1,
      });

      // If the error has a response, return it as a Response object
      if (error.response) {
        const errorHeaders: Record<string, string> = {
          'Content-Type': 'application/json',
        };
        
        // Preserve headers even in error responses
        if (error.response.headers && error.response.headers['set-cookie']) {
          errorHeaders['set-cookie'] = error.response.headers['set-cookie'];
        }
        
        const errorData = error.response.data || { detail: 'Backend request failed' };
        const errorStatus = error.response.status || 500;
        
        // Check if this is a RAG-specific error (don't retry)
        const code = errorData?.code || errorData?.detail?.code;
        const isRagError = errorStatus === 503 && (
          code === 'RAG_NOT_INITIALIZED' || 
          code === 'RAG_WARMING' || 
          code === 'RAG_NOT_CONFIGURED'
        );
        if (isRagError) {
          console.warn('IAM Backend Request: RAG error, not retrying', { 
            path, 
            status: errorStatus, 
            code 
          });
          // Return immediately without retrying
          return new Response(JSON.stringify(errorData), {
            status: errorStatus,
            headers: errorHeaders,
          });
        }
        
        return new Response(JSON.stringify(errorData), {
          status: errorStatus,
          headers: errorHeaders,
        });
      }

      // Otherwise, return a generic error
      return new Response(
        JSON.stringify({
          detail: error.message || 'Failed to connect to backend',
        }),
        {
          status: 500,
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );
    }
  }
  
  // If we exhausted all retries, return the last error
  if (lastError) {
    const statusCode = lastError.response?.status || lastError.status || 503;
    return new Response(
      JSON.stringify({
        detail: lastError.response?.data?.detail || lastError.message || 'Service temporarily unavailable. Please try again.',
      }),
      {
        status: statusCode,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
  }
  
  // Fallback (should never reach here)
  return new Response(
    JSON.stringify({ detail: 'Unknown error occurred' }),
    { status: 500, headers: { 'Content-Type': 'application/json' } }
  );
}

/**
 * Convenience method for GET requests
 */
export async function iamBackendGet(path: string, headers?: Record<string, string>) {
  return iamBackendRequest(path, { method: 'GET', headers });
}

/**
 * Convenience method for POST requests
 */
export async function iamBackendPost(
  path: string,
  body: any,
  headers?: Record<string, string>
) {
  return iamBackendRequest(path, { method: 'POST', body, headers });
}

/**
 * Convenience method for PUT requests
 */
export async function iamBackendPut(
  path: string,
  body: any,
  headers?: Record<string, string>
) {
  return iamBackendRequest(path, { method: 'PUT', body, headers });
}

/**
 * Convenience method for DELETE requests
 */
export async function iamBackendDelete(path: string, headers?: Record<string, string>) {
  return iamBackendRequest(path, { method: 'DELETE', headers });
}

/**
 * Convenience method for PATCH requests
 */
export async function iamBackendPatch(
  path: string,
  body: any,
  headers?: Record<string, string>
) {
  return iamBackendRequest(path, { method: 'PATCH', body, headers });
}

/**
 * Get IAM identity token for backend authentication.
 * Extracted for reuse in multipart upload routes.
 * 
 * @returns Promise<string> - The IAM identity token
 */
export async function getBackendIdentityToken(): Promise<string> {
  const auth = new GoogleAuth();
  const client = await auth.getIdTokenClient(BACKEND_URL);
  
  // Generate the identity token
  const tokenResponse = await client.getIdToken();
  return tokenResponse.token;
}

/**
 * Get backend base URL.
 * Extracted for reuse in multipart upload routes.
 * 
 * @returns string - The backend base URL
 */
export function getBackendBaseUrl(): string {
  return BACKEND_URL;
}


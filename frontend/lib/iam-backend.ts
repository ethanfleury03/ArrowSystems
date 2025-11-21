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

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || 'https://arrow-rag-backend-akymgh2oxq-uc.a.run.app';

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
  } = {}
): Promise<Response> {
  const { method = 'GET', body, headers = {} } = options;

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

    // Convert the response to a standard Response object
    return new Response(JSON.stringify(response.data), {
      status: response.status,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  } catch (error: any) {
    console.error('IAM Backend Request Error:', {
      path,
      error: error.message,
      status: error.response?.status,
      data: error.response?.data,
    });

    // If the error has a response, return it as a Response object
    if (error.response) {
      return new Response(JSON.stringify(error.response.data || { detail: 'Backend request failed' }), {
        status: error.response.status || 500,
        headers: {
          'Content-Type': 'application/json',
        },
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


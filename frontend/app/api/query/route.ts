import { NextRequest, NextResponse } from 'next/server';
import { getBackendUrl } from '@/lib/backend-url';

// Query summarization configuration
const SUMMARIZE_ENABLED = process.env.ENABLE_QUERY_SUMMARIZATION !== 'false'; // Default: enabled
const SUMMARIZE_MIN_LENGTH = parseInt(process.env.QUERY_SUMMARIZE_MIN_LENGTH || '500', 10); // Default: 500 chars

/**
 * Summarize a long query using the backend summarization endpoint.
 * Only called if query exceeds min_length threshold.
 */
async function summarizeQuery(query: string, backendUrl: string, authToken?: string | null): Promise<{ summary: string; wasSummarized: boolean; contentType?: string }> {
  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }
    
    const response = await fetch(`${backendUrl}/summarize-query`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      // If summarization fails, return original query
      console.warn('Query summarization failed, using original query');
      return { summary: query, wasSummarized: false };
    }

    const data = await response.json();
    return {
      summary: data.summary || query,
      wasSummarized: data.was_summarized || false,
      contentType: data.content_type
    };
  } catch (error) {
    console.error('Query summarization error:', error);
    // Fallback to original query on error
    return { summary: query, wasSummarized: false };
  }
}

export async function POST(request: NextRequest) {
  try {
    // Detect backend URL from request hostname (for network access)
    const BACKEND_URL = getBackendUrl(request);
    
    const body = await request.json();
    let query = body.query;
    let summarizationInfo = null;
    
    // Extract JWT token from request headers (sent by frontend)
    const authToken = request.headers.get('X-Auth-Token');
    
    // Summarize long queries if enabled
    if (SUMMARIZE_ENABLED && query && query.length >= SUMMARIZE_MIN_LENGTH) {
      const result = await summarizeQuery(query, BACKEND_URL, authToken);
      query = result.summary;
      if (result.wasSummarized) {
        summarizationInfo = {
          was_summarized: true,
          content_type: result.contentType,
          original_length: body.query.length,
          summarized_length: query.length
        };
      }
    }
    
    // Update body with potentially summarized query
    const processedBody = { ...body, query };
    
    // Add timeout and better error handling
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
    
    try {
      // Extract JWT token from request headers (sent by frontend)
      const authToken = request.headers.get('X-Auth-Token');
      
      // Build headers for backend request
      const backendHeaders: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      
      // Forward Authorization header to backend if token is present
      if (authToken) {
        backendHeaders['Authorization'] = `Bearer ${authToken}`;
      }
      
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: backendHeaders,
        body: JSON.stringify(processedBody),
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const error = await response.json();
        return NextResponse.json(
          { detail: error.detail || 'Backend request failed' },
          { status: response.status }
        );
      }

      const data = await response.json();
      
      // Add summarization info to response if query was summarized
      if (summarizationInfo) {
        data.summarization_info = summarizationInfo;
      }
      
      return NextResponse.json(data);
    } catch (fetchError) {
      clearTimeout(timeoutId);
      
      // Check if it was aborted (timeout)
      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        console.error('Request timeout after 5 minutes');
        return NextResponse.json(
          { detail: 'Request timed out. The query is taking too long to process. Please try a simpler query or check backend logs.' },
          { status: 504 }
        );
      }
      
      throw fetchError; // Re-throw to outer catch
    }
  } catch (error) {
    console.error('API route error:', error);
    
    // Detect backend URL for error message (fallback if detection failed)
    let backendUrlForError = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    try {
      backendUrlForError = getBackendUrl(request);
    } catch {
      // Use default if detection fails
    }
    
    // Check if it's a network error (backend not reachable)
    if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('ECONNREFUSED'))) {
      return NextResponse.json(
        { detail: `Cannot connect to backend at ${backendUrlForError}. Please check your backend URL configuration.` },
        { status: 503 }
      );
    }
    
    // Check for timeout
    if (error instanceof Error && error.name === 'AbortError') {
      return NextResponse.json(
        { detail: 'Request timed out. The query is taking too long to process.' },
        { status: 504 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}


import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';

// Query summarization configuration
const SUMMARIZE_ENABLED = process.env.ENABLE_QUERY_SUMMARIZATION !== 'false'; // Default: enabled
const SUMMARIZE_MIN_LENGTH = parseInt(process.env.QUERY_SUMMARIZE_MIN_LENGTH || '500', 10); // Default: 500 chars

/**
 * Summarize a long query using the backend summarization endpoint.
 * Only called if query exceeds min_length threshold.
 */
async function summarizeQuery(query: string, authToken?: string | null): Promise<{ summary: string; wasSummarized: boolean; contentType?: string }> {
  try {
    const headers: Record<string, string> = {};
    if (authToken) {
      headers['Authorization'] = `Bearer ${authToken}`;
    }
    
    const response = await iamBackendPost('/summarize-query', { query }, headers);

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
    const body = await request.json();
    let query = body.query;
    let summarizationInfo = null;
    
    // Extract JWT token from request headers (sent by frontend)
    const authToken = request.headers.get('X-Auth-Token');
    
    // Summarize long queries if enabled
    if (SUMMARIZE_ENABLED && query && query.length >= SUMMARIZE_MIN_LENGTH) {
      const result = await summarizeQuery(query, authToken);
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
    
    try {
      // Build headers for backend request
      const backendHeaders: Record<string, string> = {};
      
      // Forward Authorization header to backend if token is present
      if (authToken) {
        backendHeaders['Authorization'] = `Bearer ${authToken}`;
      }
      
      const response = await iamBackendPost('/query', processedBody, backendHeaders);

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
      // Check if it was a timeout or other error
      if (fetchError instanceof Error) {
        console.error('Request error:', fetchError.message);
        return NextResponse.json(
          { detail: fetchError.message || 'Request failed' },
          { status: 504 }
        );
      }
      
      throw fetchError; // Re-throw to outer catch
    }
  } catch (error) {
    console.error('API route error:', error);
    
    // Check if it's a network error (backend not reachable)
    if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('ECONNREFUSED'))) {
      return NextResponse.json(
        { detail: 'Cannot connect to backend. Please check your backend configuration.' },
        { status: 503 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}


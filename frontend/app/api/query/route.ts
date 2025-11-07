import { NextRequest, NextResponse } from 'next/server';

// Use BACKEND_URL from env (set in Docker) or default to localhost for local dev
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Query summarization configuration
const SUMMARIZE_ENABLED = process.env.ENABLE_QUERY_SUMMARIZATION !== 'false'; // Default: enabled
const SUMMARIZE_MIN_LENGTH = parseInt(process.env.QUERY_SUMMARIZE_MIN_LENGTH || '500', 10); // Default: 500 chars

/**
 * Summarize a long query using the backend summarization endpoint.
 * Only called if query exceeds min_length threshold.
 */
async function summarizeQuery(query: string): Promise<{ summary: string; wasSummarized: boolean; contentType?: string }> {
  try {
    const response = await fetch(`${BACKEND_URL}/summarize-query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
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
    const body = await request.json();
    let query = body.query;
    let summarizationInfo = null;
    
    // Summarize long queries if enabled
    if (SUMMARIZE_ENABLED && query && query.length >= SUMMARIZE_MIN_LENGTH) {
      const result = await summarizeQuery(query);
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
      const response = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
    
    // Check if it's a network error (backend not reachable)
    if (error instanceof TypeError && (error.message.includes('fetch') || error.message.includes('ECONNREFUSED'))) {
      return NextResponse.json(
        { detail: `Cannot connect to backend at ${BACKEND_URL}. Make sure the backend is running on port 8000.` },
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


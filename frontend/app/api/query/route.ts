import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

// Query summarization configuration
const SUMMARIZE_ENABLED = process.env.ENABLE_QUERY_SUMMARIZATION !== 'false'; // Default: enabled
const SUMMARIZE_MIN_LENGTH = parseInt(process.env.QUERY_SUMMARIZE_MIN_LENGTH || '500', 10); // Default: 500 chars

/**
 * Summarize a long query using the backend summarization endpoint.
 * Only called if query exceeds min_length threshold.
 *
 * The user JWT is forwarded via X-User-Token so the backend can
 * associate summarization requests with the current user.
 */
async function summarizeQuery(
  query: string,
  userToken?: string | null,
): Promise<{ summary: string; wasSummarized: boolean; contentType?: string }> {
  try {
    const headers: Record<string, string> = {};
    if (userToken) {
      headers['X-User-Token'] = userToken;
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
      contentType: data.content_type,
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

    // Validate and normalize incoming body from client
    const rawQuery = body.query;
    if (!rawQuery || typeof rawQuery !== 'string' || !rawQuery.trim()) {
      return NextResponse.json(
        { detail: 'Query is required' },
        { status: 400 },
      );
    }

    // Extract JWT token from HTTP-only cookie
    const userToken = await extractJwtFromCookie();
    if (!userToken) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 },
      );
    }

    let query = rawQuery as string;
    let summarizationInfo: {
      was_summarized: boolean;
      content_type?: string;
      original_length: number;
      summarized_length: number;
    } | null = null;

    // Summarize long queries if enabled
    if (SUMMARIZE_ENABLED && query && query.length >= SUMMARIZE_MIN_LENGTH) {
      const result = await summarizeQuery(query, userToken);
      query = result.summary;
      if (result.wasSummarized) {
        summarizationInfo = {
          was_summarized: true,
          content_type: result.contentType,
          original_length: rawQuery.length,
          summarized_length: query.length,
        };
      }
    }

    // Map client body to backend QueryRequest shape
    const {
      top_k,
      alpha,
      dynamic_windowing,
      metadata_filters,
      machine_confirmation,
      selected_machine,
      session_id,
    } = body;

    const backendBody: Record<string, any> = {
      query,
    };

    // Optional fields - only include when valid so backend defaults still work
    if (typeof session_id === 'string' && session_id.trim().length > 0) {
      backendBody.session_id = session_id;
    }

    if (top_k !== undefined && top_k !== null && top_k !== '') {
      const parsedTopK = Number(top_k);
      if (!Number.isNaN(parsedTopK)) {
        backendBody.top_k = parsedTopK;
      }
    }

    if (alpha !== undefined && alpha !== null && alpha !== '') {
      const parsedAlpha = Number(alpha);
      if (!Number.isNaN(parsedAlpha)) {
        backendBody.alpha = parsedAlpha;
      }
    }

    if (typeof dynamic_windowing === 'boolean') {
      backendBody.dynamic_windowing = dynamic_windowing;
    }

    if (metadata_filters && typeof metadata_filters === 'object') {
      backendBody.metadata_filters = metadata_filters;
    }

    if (typeof machine_confirmation === 'boolean') {
      backendBody.machine_confirmation = machine_confirmation;
    }

    if (typeof selected_machine === 'string' && selected_machine.trim().length > 0) {
      backendBody.selected_machine = selected_machine;
    }

    try {
      // Call backend /query with user JWT in X-User-Token header
      const response = await iamBackendPost('/query', backendBody, {
        'X-User-Token': userToken,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Backend request failed' }));
        return NextResponse.json(
          { detail: error.detail || 'Backend request failed' },
          { status: response.status },
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
          { status: 504 },
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
        { status: 503 },
      );
    }

    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 },
    );
  }
}


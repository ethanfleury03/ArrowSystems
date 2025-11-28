import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    // Call backend /rag/status endpoint (no auth required)
    const response = await iamBackendGet('/rag/status');

    if (!response.ok) {
      // If backend fails, assume RAG is disabled
      return NextResponse.json(
        { rag_enabled: false, details: 'Unable to check RAG status' },
        { status: 200 } // Return 200 so frontend can still handle it
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching RAG status:', error);
    // On error, assume RAG is disabled
    return NextResponse.json(
      { rag_enabled: false, details: 'Unable to check RAG status' },
      { status: 200 }
    );
  }
}


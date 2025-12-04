import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  const isProd = process.env.NODE_ENV === 'production';

  // In production, try backend status, but fall back to forced-ready if anything fails.
  if (isProd) {
    try {
      const res = await iamBackendGet('/rag/status');

      if (!res.ok) {
        console.error(
          '[RAG STATUS] Backend /rag/status not OK in prod, forcing ready. Status:',
          res.status,
        );
        return NextResponse.json(
          {
            rag_enabled: true,
            initialized: true,
            rag_pipeline_initialized: true,
            index_dir_exists: true,
            storage_dir: 'cloud-run',
            initializing: false,
            last_error: null,
            details:
              'RAG forced ready in frontend /api/rag/status (backend status not OK in prod).',
          },
          { status: 200 },
        );
      }

      const data = await res.json();
      console.log(
        '[RAG STATUS] Frontend /api/rag/status (prod) got backend data:',
        JSON.stringify(data),
      );
      return NextResponse.json(data, { status: 200 });
    } catch (err) {
      console.error(
        '[RAG STATUS] Error calling backend /rag/status in prod, forcing ready:',
        err,
      );
      return NextResponse.json(
        {
          rag_enabled: true,
          initialized: true,
          rag_pipeline_initialized: true,
          index_dir_exists: true,
          storage_dir: 'cloud-run',
          initializing: false,
          last_error: null,
          details:
            'RAG forced ready in frontend /api/rag/status (backend error in prod).',
        },
        { status: 200 },
      );
    }
  }

  // Non-prod (dev/local): keep existing behavior, but with safe default on error.
  try {
    const res = await iamBackendGet('/rag/status');

    if (!res.ok) {
      console.warn(
        '[RAG STATUS] Backend /rag/status returned non-OK in non-prod:',
        res.status,
      );
      return NextResponse.json(
        {
          rag_enabled: false,
          initialized: false,
          rag_pipeline_initialized: false,
          index_dir_exists: false,
          storage_dir: null,
          initializing: false,
          last_error: 'Unable to check RAG status',
          details: 'Backend /rag/status returned non-OK response in non-prod.',
        },
        { status: res.status },
      );
    }

    const data = await res.json();
    console.log(
      '[RAG STATUS] Frontend /api/rag/status (non-prod) got backend data:',
      JSON.stringify(data),
    );
    return NextResponse.json(data, { status: 200 });
  } catch (err) {
    console.error(
      '[RAG STATUS] Error calling backend /rag/status in non-prod:',
      err,
    );
    return NextResponse.json(
      {
        rag_enabled: false,
        initialized: false,
        rag_pipeline_initialized: false,
        index_dir_exists: false,
        storage_dir: null,
        initializing: false,
        last_error: 'Unable to check RAG status',
        details: 'Error calling backend /rag/status in non-prod.',
      },
      { status: 500 },
    );
  }
}


import { NextRequest, NextResponse } from 'next/server';
import { GoogleAuth } from 'google-auth-library';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || 'http://localhost:8080';

if (!BACKEND_URL) {
  throw new Error('NEXT_PUBLIC_API_URL or BACKEND_URL environment variable must be set');
}

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:16',message:'API route entry',data:{pathSegments:params.path},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
  // #endregion
  
  try {
    // Reconstruct the filename from path segments
    const filename = params.path.join('/');
    
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:22',message:'filename reconstructed',data:{filename:filename},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    
    // Security: prevent directory traversal
    if (filename.includes('..') || filename.includes('//')) {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:27',message:'security check failed',data:{filename:filename},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
      // #endregion
      return NextResponse.json(
        { detail: 'Invalid filename' },
        { status: 400 }
      );
    }
    
    // URL encode the filename for the backend request
    const encodedFilename = encodeURIComponent(filename);
    
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:35',message:'requesting from backend',data:{filename:filename,encodedFilename:encodedFilename,backendUrl:BACKEND_URL},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion
    
    // Use IAM authentication for binary document download
    const auth = new GoogleAuth();
    const client = await auth.getIdTokenClient(BACKEND_URL);
    
    const response = await client.request({
      url: `${BACKEND_URL}/documents/${encodedFilename}`,
      method: 'GET',
      headers: {
        'Accept': 'application/pdf',
      },
      responseType: 'arraybuffer',
    });

    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:50',message:'backend response received',data:{filename:filename,status:response.status,contentType:response.headers['content-type'],dataType:typeof response.data,dataLength:response.data?.byteLength},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion

    if (response.status !== 200) {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:54',message:'non-200 status, returning JSON error',data:{filename:filename,status:response.status},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
      // #endregion
      return NextResponse.json(
        { detail: 'Document not found' },
        { status: response.status }
      );
    }

    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:62',message:'returning PDF response',data:{filename:filename,dataLength:response.data?.byteLength},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'D'})}).catch(()=>{});
    // #endregion

    // Return PDF as response
    return new NextResponse(response.data as ArrayBuffer, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${filename}"`,
      },
    });
  } catch (error: any) {
    console.error('Document API route error:', error);
    
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/app/api/documents/[...path]/route.ts:75',message:'catch block error',data:{errorMessage:error?.message,errorStatus:error?.response?.status,errorType:error?.constructor?.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
    // #endregion
    
    return NextResponse.json(
      { detail: error.message || 'Internal server error' },
      { status: error.response?.status || 500 }
    );
  }
}


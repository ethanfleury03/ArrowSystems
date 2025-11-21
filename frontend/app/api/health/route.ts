import { NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET() {
  try {
    const response = await iamBackendGet('/health');
    if (!response.ok) {
      throw new Error(`Backend health check failed with status ${response.status}`);
    }
    const backendHealth = await response.json();
    return NextResponse.json({
      status: 'ok',
      backend: backendHealth,
      timestamp: new Date().toISOString(),
      environment: process.env.NODE_ENV || 'development',
    });
  } catch (error) {
    console.error('Health check failed:', error);
    return NextResponse.json(
      {
        status: 'error',
        timestamp: new Date().toISOString(),
        environment: process.env.NODE_ENV || 'development',
        error: process.env.NODE_ENV === 'development' ? String(error) : undefined,
      },
      { status: 503 }
    );
  }
}

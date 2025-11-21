import { NextRequest, NextResponse } from 'next/server';
import { iamBackendDelete } from '@/lib/iam-backend';

export async function DELETE(
  request: NextRequest,
  { params }: { params: { chunkId: string } }
) {
  try {
    const response = await iamBackendDelete(`/admin/chunks/${params.chunkId}`);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to delete chunk' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin chunk delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}


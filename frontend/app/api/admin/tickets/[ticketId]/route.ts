import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPatch, iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function GET(
  request: NextRequest,
  { params }: { params: { ticketId: string } }
) {
  try {
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Authentication required' },
        { status: 401 }
      );
    }

    const ticketId = params.ticketId;

    const response = await iamBackendGet(
      `/admin/tickets/${ticketId}`,
      { 'X-User-Token': token }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch ticket details' }));
      return NextResponse.json(
        errorData,
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error fetching ticket details:', error);
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { ticketId: string } }
) {
  try {
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Authentication required' },
        { status: 401 }
      );
    }

    const body = await request.json();
    const ticketId = params.ticketId;

    const response = await iamBackendPatch(
      `/admin/tickets/${ticketId}`,
      body,
      { 'X-User-Token': token }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to update ticket' }));
      return NextResponse.json(
        errorData,
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error updating ticket:', error);
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
}

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

/**
 * Middleware to handle authentication and route protection.
 * 
 * Protected routes (require authentication):
 * - / (root/chat)
 * - /admin/*
 * - /account
 * 
 * Public routes (no auth required):
 * - /login
 * - /register
 * - /api/auth/*
 * - /api/health
 * - /api/rag/status (for status checks)
 * 
 * Note: This middleware checks for cookie presence only.
 * Actual token validation happens in page components/layouts.
 */
export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  
  // Public routes that don't require authentication
  const publicRoutes = [
    '/login',
    '/register',
    '/api/auth/login',
    '/api/auth/register',
    '/api/auth/logout',
    '/api/health',
    '/api/rag/status',
  ];
  
  // Check if the current path is a public route
  const isPublicRoute = publicRoutes.some(route => pathname.startsWith(route));
  
  // If it's a public route, allow access
  // (Pages will handle redirecting authenticated users away from /login)
  if (isPublicRoute) {
    return NextResponse.next();
  }
  
  // Protected routes - check for authentication token
  const token = request.cookies.get('access_token');
  
  if (!token) {
    // No token, redirect to login
    const loginUrl = new URL('/login', request.url);
    // Preserve the original URL as a redirect parameter
    loginUrl.searchParams.set('redirect', pathname);
    return NextResponse.redirect(loginUrl);
  }
  
  // Token exists - allow access (pages will validate token validity)
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files (public folder)
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};

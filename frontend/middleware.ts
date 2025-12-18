import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getAuthCookieName } from './lib/auth-config';

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
  
  // Early bypass for static assets and ACME challenge paths
  // These must NEVER be redirected for TLS provisioning and SEO
  if (
    pathname.startsWith('/.well-known/') ||
    pathname === '/robots.txt' ||
    pathname === '/sitemap.xml' ||
    pathname === '/favicon.ico' ||
    pathname.startsWith('/_next/')
  ) {
    return NextResponse.next();
  }
  
  // Public routes that don't require authentication
  const publicRoutes = [
    '/login',
    '/register',
    '/accept-invite', // Invite password setup page
    '/api/auth/login',
    '/api/auth/register',
    '/api/auth/logout',
    '/api/auth/invite/validate', // Invite validation endpoint
    '/api/auth/invite/accept', // Invite acceptance endpoint
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
  const cookieName = getAuthCookieName();
  const token = request.cookies.get(cookieName);
  
  if (!token) {
    // For API routes, NEVER redirect with HTML (breaks callers expecting JSON).
    // Return JSON 401 instead so the UI can handle it safely.
    if (pathname.startsWith('/api/')) {
      return NextResponse.json({ detail: 'Not authenticated' }, { status: 401 });
    }

    // For pages, redirect to login.
    const loginUrl = new URL('/login', request.url);
    const fullPath = request.nextUrl.pathname + request.nextUrl.search;
    loginUrl.searchParams.set('redirect', fullPath);
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
     * - _next/ (all Next.js internal routes)
     * - .well-known/ (ACME challenges, security.txt, etc.)
     * - favicon.ico (favicon file)
     * - robots.txt, sitemap.xml (SEO files)
     * - public files (public folder - images, etc.)
     */
    '/((?!_next/|\\.well-known/|favicon\\.ico|robots\\.txt|sitemap\\.xml|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};

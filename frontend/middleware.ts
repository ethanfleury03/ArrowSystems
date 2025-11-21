import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getAuthCookieName } from './lib/auth-config';

const disableAuth =
  process.env.DISABLE_AUTH === 'true' ||
  process.env.NEXT_PUBLIC_DISABLE_AUTH === 'true';

export function middleware(request: NextRequest) {
  if (disableAuth) {
    return NextResponse.next();
  }

  // Get the JWT auth cookie
  const authCookieName = getAuthCookieName();
  const authCookie = request.cookies.get(authCookieName);
  const pathname = request.nextUrl.pathname;

  // Redirect /register to /login (registration is disabled)
  if (pathname === '/register') {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // Allow access to login page
  if (pathname === '/login') {
    if (authCookie) {
      // Redirect to home if already logged in
      return NextResponse.redirect(new URL('/', request.url));
    }
    // Allow access to login page - don't redirect if already on login
    return NextResponse.next();
  }

  // Protect all other routes (including root)
  if (!authCookie) {
    // Only redirect to login if not already going there (prevent redirect loops)
    // Clean the URL to avoid query parameters that might cause issues
    const loginUrl = new URL('/login', request.url);
    loginUrl.search = ''; // Remove any query parameters
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files (public folder)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};


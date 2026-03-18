import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

/**
 * Middleware to handle socket.io polling requests
 * 
 * Socket.io client tries to connect to /socket.io by default.
 * The actual socket.io server runs on port 3003 (training-service).
 * 
 * This middleware intercepts /socket.io requests and returns a proper response
 * to avoid 404 errors flooding the dev server logs.
 */
export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Handle socket.io polling requests
  if (pathname === '/socket.io' || pathname.startsWith('/socket.io/')) {
    // Return a simple response to avoid 404
    return new NextResponse(
      JSON.stringify({
        message: 'Socket.io endpoint',
        hint: 'Use XTransformPort=3003 to connect to the socket.io server'
      }),
      {
        status: 200,
        headers: {
          'Content-Type': 'application/json',
          'Access-Control-Allow-Origin': '*',
        }
      }
    )
  }

  // Continue with normal request handling
  return NextResponse.next()
}

// Configure which routes this middleware applies to
export const config = {
  matcher: [
    /*
     * Match socket.io routes
     */
    '/socket.io',
    '/socket.io/:path*',
  ],
}

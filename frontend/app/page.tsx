import { ChatInterface } from "@/components/chat-interface"
import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'

/**
 * Root page - requires authentication
 * 
 * Auth behavior:
 * - If not authenticated: redirect to /login
 * - If authenticated: show ChatInterface
 * 
 * This ensures users are not silently auto-logged in from previous sessions.
 * The middleware also protects this route, but this provides an additional check.
 */
export default async function Home() {
  // Check for auth token cookie
  const cookieStore = await cookies()
  const token = cookieStore.get('access_token')
  
  // If no token, redirect to login
  // Note: Middleware should also catch this, but this provides server-side enforcement
  if (!token) {
    redirect('/login')
  }
  
  // Token exists - middleware will validate it
  // If invalid, middleware will redirect to login
  // If valid, show chat interface
  return (
    <main className="flex min-h-screen flex-col">
      <ChatInterface />
    </main>
  )
}


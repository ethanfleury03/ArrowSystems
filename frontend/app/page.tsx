import { ChatInterface } from "@/components/chat-interface"
import { cookies } from 'next/headers'
import { redirect } from 'next/navigation'
import { getAuthCookieName } from '@/lib/auth-config'

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
  // Check for auth token cookie using configured name
  const cookieStore = await cookies()
  const cookieName = getAuthCookieName()
  const token = cookieStore.get(cookieName)
  
  // If no token or empty value, redirect to login immediately (server-side, no UI flash)
  if (!token || !token.value || token.value.trim() === '') {
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


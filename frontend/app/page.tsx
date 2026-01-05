import { ChatInterface } from "@/components/chat-interface"

/**
 * Root page - requires authentication
 * 
 * Auth behavior:
 * - Middleware protects this route (checks cookie presence)
 * - ChatInterface component handles client-side auth gating (tri-state: loading/guest/authed)
 * - No server-side cookie check needed - middleware already handles it
 * 
 * This keeps page transitions fast by avoiding blocking server-side cookie reads.
 */
export default function Home() {
  // Middleware already protects this route and checks for cookie
  // ChatInterface component handles client-side auth state and redirects if needed
  return (
    <main className="flex min-h-screen flex-col">
      <ChatInterface />
    </main>
  )
}


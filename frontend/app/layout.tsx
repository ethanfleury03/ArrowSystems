import type { Metadata } from 'next'
import './globals.css'
import { ErrorBoundary } from '@/components/ErrorBoundary'

export const metadata: Metadata = {
  title: 'RAG Assistant',
  description: 'AI-powered RAG Assistant for technical documentation',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* Prevent FOUC (Flash of Unstyled Content) */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Prevent FOUC by hiding body until CSS loads
              (function() {
                try {
                  var html = document.documentElement;
                  html.style.visibility = 'hidden';
                  window.addEventListener('load', function() {
                    html.style.visibility = 'visible';
                  });
                  // Fallback: show after 1 second even if load event doesn't fire
                  setTimeout(function() {
                    html.style.visibility = 'visible';
                  }, 1000);
                } catch(e) {}
              })();
            `,
          }}
        />
      </head>
      <body>
        <ErrorBoundary>
          {children}
        </ErrorBoundary>
      </body>
    </html>
  )
}


'use client';

import React, { Component, ErrorInfo, ReactNode } from 'react';
import { isDev } from '@/lib/env';
import { logger } from '@/lib/logger';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

/**
 * React Error Boundary component
 * Catches JavaScript errors in child components and displays fallback UI
 * 
 * Dev: Shows full error details and stack trace
 * Prod: Shows user-friendly error message, logs full details
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    // Log error details
    logger.error('ErrorBoundary caught an error', error, {
      componentStack: errorInfo.componentStack,
    });

    // In dev mode, also log to console for easier debugging
    if (isDev) {
      console.error('ErrorBoundary caught an error:', error);
      console.error('Error info:', errorInfo);
    }
  }

  render(): ReactNode {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      if (isDev) {
        // Development: Show full error details
        return (
          <div className="min-h-screen flex items-center justify-center bg-red-50 p-4">
            <div className="max-w-2xl w-full bg-white rounded-lg shadow-lg p-6 border-2 border-red-500">
              <h1 className="text-2xl font-bold text-red-600 mb-4">
                ⚠️ Error Boundary Caught an Error
              </h1>
              {this.state.error && (
                <div className="space-y-4">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-800 mb-2">Error Message:</h2>
                    <pre className="bg-gray-100 p-3 rounded text-sm text-red-600 overflow-auto">
                      {this.state.error.message}
                    </pre>
                  </div>
                  {this.state.error.stack && (
                    <div>
                      <h2 className="text-lg font-semibold text-gray-800 mb-2">Stack Trace:</h2>
                      <pre className="bg-gray-100 p-3 rounded text-xs text-gray-700 overflow-auto max-h-96">
                        {this.state.error.stack}
                      </pre>
                    </div>
                  )}
                  <button
                    onClick={() => {
                      this.setState({ hasError: false, error: null });
                      window.location.reload();
                    }}
                    className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                  >
                    Reload Page
                  </button>
                </div>
              )}
            </div>
          </div>
        );
      } else {
        // Production: Show user-friendly error message
        return (
          <div className="min-h-screen flex items-center justify-center bg-gray-50 p-4">
            <div className="max-w-md w-full bg-white rounded-lg shadow-lg p-6 text-center">
              <div className="text-6xl mb-4">😕</div>
              <h1 className="text-2xl font-bold text-gray-800 mb-2">
                Something went wrong
              </h1>
              <p className="text-gray-600 mb-6">
                We&apos;re sorry, but something unexpected happened. Please try refreshing the page.
              </p>
              <button
                onClick={() => {
                  this.setState({ hasError: false, error: null });
                  window.location.reload();
                }}
                className="px-6 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
              >
                Reload Page
              </button>
            </div>
          </div>
        );
      }
    }

    return this.props.children;
  }
}








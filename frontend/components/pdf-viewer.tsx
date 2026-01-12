"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut } from "lucide-react"
import { cn } from "@/lib/utils"

interface PDFViewerProps {
  filename: string
  initialPage?: number
  onClose: () => void
}

export function PDFViewer({ filename, initialPage = 1, onClose }: PDFViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [currentPage, setCurrentPage] = useState(initialPage)
  const [numPages, setNumPages] = useState<number | null>(null)
  const [scale, setScale] = useState(1.0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Load PDF using iframe (simple approach)
    // In production, you might want to use react-pdf or pdf.js
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/components/pdf-viewer.tsx:25',message:'PDFViewer useEffect triggered',data:{filename:filename,initialPage:initialPage},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
    // #endregion
    setLoading(true)
    setError(null)
  }, [filename])

  const handlePreviousPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1)
    }
  }

  const handleNextPage = () => {
    if (numPages && currentPage < numPages) {
      setCurrentPage(currentPage + 1)
    }
  }

  const handleZoomIn = () => {
    setScale(Math.min(scale + 0.25, 3.0))
  }

  const handleZoomOut = () => {
    setScale(Math.max(scale - 0.25, 0.5))
  }

  // Use Next.js API route to proxy PDF requests
  const pdfUrl = `/api/documents/${encodeURIComponent(filename)}#page=${currentPage}`
  
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/components/pdf-viewer.tsx:52',message:'PDF URL constructed',data:{filename:filename,pdfUrl:pdfUrl,currentPage:currentPage},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
  // #endregion

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80">
      <div className="relative flex h-full w-full flex-col bg-background">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold truncate max-w-md">{filename}</h2>
            {numPages && (
              <span className="text-sm text-muted-foreground">
                Page {currentPage} of {numPages}
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 border-r border-border pr-2 mr-2">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleZoomOut}
                disabled={scale <= 0.5}
              >
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-xs text-muted-foreground min-w-[3rem] text-center">
                {Math.round(scale * 100)}%
              </span>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleZoomIn}
                disabled={scale >= 3.0}
              >
                <ZoomIn className="h-4 w-4" />
              </Button>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handlePreviousPage}
              disabled={currentPage <= 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleNextPage}
              disabled={numPages ? currentPage >= numPages : true}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* PDF Content */}
        <div ref={containerRef} className="flex-1 overflow-auto bg-gray-100 dark:bg-gray-900">
          {error ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <p className="text-destructive mb-2">Failed to load PDF</p>
                <p className="text-sm text-muted-foreground">{error}</p>
              </div>
            </div>
          ) : (
            <div className="flex h-full items-center justify-center p-4">
              <iframe
                src={pdfUrl}
                className="w-full h-full border-0"
                style={{ transform: `scale(${scale})`, transformOrigin: 'top left' }}
                onLoad={() => {
                  // #region agent log
                  fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/components/pdf-viewer.tsx:127',message:'iframe onLoad triggered',data:{filename:filename,pdfUrl:pdfUrl},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                  // #endregion
                  setLoading(false)
                  // Try to get page count from iframe (may not work with all PDFs)
                  // For now, we'll just show the PDF
                }}
                onError={() => {
                  // #region agent log
                  fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/components/pdf-viewer.tsx:133',message:'iframe onError triggered',data:{filename:filename,pdfUrl:pdfUrl},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
                  // #endregion
                  setError("Failed to load PDF. Please check if the file exists.")
                  setLoading(false)
                }}
              />
            </div>
          )}
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/80">
              <div className="text-center">
                <div className="mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto" />
                <p className="text-sm text-muted-foreground">Loading PDF...</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


"use client"

import { DocumentSource } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { FileText, ExternalLink } from "lucide-react"
import { PDFViewer } from "./pdf-viewer"
import { useState } from "react"

interface DocumentsPanelProps {
  documentSources: DocumentSource[]
}

export function DocumentsPanel({ documentSources }: DocumentsPanelProps) {
  const [selectedDoc, setSelectedDoc] = useState<{ filename: string; page?: number } | null>(null)

  if (!documentSources || documentSources.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center p-8 text-center">
        <FileText className="mb-4 h-12 w-12 text-muted-foreground" />
        <h3 className="mb-2 text-lg font-semibold">No Documents</h3>
        <p className="text-sm text-muted-foreground">
          No documents were used as sources for this query.
        </p>
      </div>
    )
  }

  return (
    <>
      <div className="flex h-full flex-col overflow-y-auto p-4">
        <div className="mb-4">
          <h2 className="text-lg font-semibold mb-1">Source Documents</h2>
          <p className="text-sm text-muted-foreground">
            {documentSources.length} document{documentSources.length !== 1 ? 's' : ''} used to generate this answer
          </p>
        </div>

        <div className="space-y-3">
          {documentSources.map((doc, index) => {
            const hasPages = doc.pages_used && doc.pages_used.length > 0
            const firstPage = hasPages ? doc.pages_used[0] : undefined

            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/components/documents-panel.tsx:43',message:'document source mapped',data:{doc_id:doc.doc_id,index:index,hasPages:hasPages,firstPage:firstPage},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
            // #endregion

            return (
              <Card
                key={index}
                className="group cursor-pointer transition-all hover:border-primary hover:shadow-md"
                onClick={() => {
                  // #region agent log
                  fetch('http://127.0.0.1:7242/ingest/33e5f654-3cb0-435b-825c-00380806eaa2',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'frontend/components/documents-panel.tsx:50',message:'card clicked, setting selectedDoc',data:{doc_id:doc.doc_id,firstPage:firstPage},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'E'})}).catch(()=>{});
                  // #endregion
                  setSelectedDoc({ filename: doc.doc_id, page: firstPage })
                }}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 min-w-0">
                      <CardTitle className="text-base truncate">{doc.doc_id}</CardTitle>
                      {hasPages && (
                        <CardDescription className="mt-1">
                          Pages: {doc.pages_used.join(', ')}
                        </CardDescription>
                      )}
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedDoc({ filename: doc.doc_id, page: firstPage })
                      }}
                    >
                      <ExternalLink className="h-4 w-4" />
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="pt-0 space-y-3">
                  {doc.snippet && (
                    <p className="text-sm text-muted-foreground line-clamp-3 leading-relaxed">
                      {doc.snippet}
                    </p>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full"
                    onClick={(e) => {
                      e.stopPropagation()
                      setSelectedDoc({ filename: doc.doc_id, page: firstPage })
                    }}
                  >
                    <FileText className="mr-2 h-4 w-4" />
                    View Document
                    {hasPages && firstPage && ` (Page ${firstPage})`}
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {selectedDoc && (
        <PDFViewer
          filename={selectedDoc.filename}
          initialPage={selectedDoc.page || 1}
          onClose={() => setSelectedDoc(null)}
        />
      )}
    </>
  )
}


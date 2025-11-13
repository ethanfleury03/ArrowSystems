"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { resolveApiBaseUrl } from "@/config/api";
import { Upload, FileText, Trash2, Edit, Eye, EyeOff, X, Check, ExternalLink } from "lucide-react";

interface Document {
  filename: string;
  size_bytes: number;
  uploaded_date?: string | null;
  chunk_count: number;
  page_count: number;
  file_path: string;
  file_type: string;
  is_active: boolean;
  machine_model?: string | null;
  category?: string | null;
  product_family?: string | null;
}

type SortField = keyof Pick<Document, "filename" | "chunk_count" | "page_count" | "is_active">;
type SortDirection = "asc" | "desc";

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

export default function AdminDocumentsPage() {
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loadingTable, setLoadingTable] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<SortField>("filename");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);

  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState<string>("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [actionSubmitting, setActionSubmitting] = useState(false);

  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [toastType, setToastType] = useState<"success" | "error">("success");

  const apiBaseUrl = useMemo(() => resolveApiBaseUrl(), []);

  const showToast = useCallback((message: string, type: "success" | "error" = "success") => {
    setToastType(type);
    setToastMessage(message);
    window.setTimeout(() => setToastMessage(null), 3000);
  }, []);

  const extractApiError = (detail: unknown): string | null => {
    if (!detail) return null;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((entry) => {
          if (typeof entry === "string") return entry;
          if (entry && typeof entry === "object") {
            const msg = (entry as Record<string, unknown>).msg;
            const loc = (entry as Record<string, unknown>).loc;
            if (msg && typeof msg === "string") {
              if (Array.isArray(loc)) {
                return `${msg} (${loc.join(" → ")})`;
              }
              return msg;
            }
          }
          try {
            return JSON.stringify(entry);
          } catch (error) {
            return String(entry);
          }
        })
        .join("; ");
    }
    if (typeof detail === "object") {
      const nested = (detail as Record<string, unknown>).detail;
      if (nested && nested !== detail) {
        return extractApiError(nested);
      }
      try {
        return JSON.stringify(detail);
      } catch (error) {
        return String(detail);
      }
    }
    return String(detail);
  };

  const fetchDocuments = useCallback(
    async (token: string) => {
      setLoadingTable(true);
      setError(null);
      try {
        const response = await fetch(`${apiBaseUrl}/admin/documents`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!response.ok) {
          throw new Error(`Failed to load documents (${response.status})`);
        }
        const data = await response.json();
        setDocuments(Array.isArray(data.documents) ? data.documents : []);
      } catch (err) {
        console.error("Failed to fetch documents:", err);
        setError(err instanceof Error ? err.message : "Unable to load documents.");
      } finally {
        setLoadingTable(false);
      }
    },
    [apiBaseUrl]
  );

  useEffect(() => {
    try {
      const token = localStorage.getItem("auth_token");
      if (token) {
        setAuthToken(token);
        fetchDocuments(token);
      }
    } catch (error) {
      console.warn("Failed to retrieve auth token:", error);
      setError("Unable to access authentication token.");
    }
  }, [fetchDocuments]);

  const filteredDocuments = useMemo(() => {
    const term = searchTerm.trim().toLowerCase();
    if (!term) return documents;
    return documents.filter((doc) => {
      return doc.filename.toLowerCase().includes(term);
    });
  }, [documents, searchTerm]);

  const sortedDocuments = useMemo(() => {
    const sorted = [...filteredDocuments];
    sorted.sort((a, b) => {
      const direction = sortDirection === "asc" ? 1 : -1;
      const aValue = a[sortField];
      const bValue = b[sortField];
      if (typeof aValue === "number" && typeof bValue === "number") {
        return (aValue - bValue) * direction;
      }
      if (typeof aValue === "boolean" && typeof bValue === "boolean") {
        return (aValue === bValue ? 0 : aValue ? 1 : -1) * direction;
      }
      return String(aValue).localeCompare(String(bValue)) * direction;
    });
    return sorted;
  }, [filteredDocuments, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const resetFormState = () => {
    setSelectedDocument(null);
    setUploadFile(null);
    setUploadProgress("");
    setDeleteConfirmation("");
  };

  const closeAllModals = () => {
    setIsUploadModalOpen(false);
    setIsEditModalOpen(false);
    setIsDeleteModalOpen(false);
    setIsViewModalOpen(false);
    resetFormState();
  };

  const handleUpload = () => {
    resetFormState();
    setIsUploadModalOpen(true);
  };

  const handleEdit = (doc: Document) => {
    setSelectedDocument(doc);
    setIsEditModalOpen(true);
  };

  const handleDelete = (doc: Document) => {
    setSelectedDocument(doc);
    setDeleteConfirmation("");
    setIsDeleteModalOpen(true);
  };

  const handleViewDocument = (doc: Document) => {
    setSelectedDocument(doc);
    setIsViewModalOpen(true);
  };

  const handleToggleActive = async (doc: Document) => {
    if (!authToken) return;
    setActionSubmitting(true);
    try {
      const encodedFilename = encodeURIComponent(doc.filename);
      const response = await fetch(`${apiBaseUrl}/admin/documents/${encodedFilename}/toggle`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ is_active: !doc.is_active }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to toggle document status");
      }
      showToast(`✅ Document ${!doc.is_active ? "enabled" : "disabled"}`);
      await fetchDocuments(authToken);
    } catch (err) {
      console.error("Toggle document status failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to toggle document status", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  const submitUpload = async () => {
    if (!authToken || !uploadFile) return;
    setActionSubmitting(true);
    setUploadProgress("Uploading file...");
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);

      setUploadProgress("Uploading file to server...");
      const response = await fetch(`${apiBaseUrl}/admin/documents/upload`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to upload document");
      }

      setUploadProgress("Ingesting document into index (this may take a moment)...");
      const result = await response.json();
      
      setUploadProgress(
        `✅ Complete! Processed ${result.page_count || 0} pages, created ${result.chunk_count || 0} chunks. Reloading index...`
      );
      
      showToast(`✅ Document uploaded and ingested successfully (${result.chunk_count || 0} chunks created)`);
      await fetchDocuments(authToken);
      
      // Small delay to show completion message
      setTimeout(() => {
        closeAllModals();
      }, 1500);
    } catch (err) {
      console.error("Upload failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to upload document", "error");
      setUploadProgress("");
    } finally {
      setActionSubmitting(false);
    }
  };


  const submitDelete = async () => {
    if (!authToken || !selectedDocument) return;
    if (deleteConfirmation !== "DELETE") {
      showToast("Please type DELETE to confirm", "error");
      return;
    }
    setActionSubmitting(true);
    try {
      const encodedFilename = encodeURIComponent(selectedDocument.filename);
      const response = await fetch(`${apiBaseUrl}/admin/documents/${encodedFilename}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${authToken}`,
        },
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to delete document");
      }
      showToast("✅ Document deleted");
      await fetchDocuments(authToken);
      closeAllModals();
    } catch (err) {
      console.error("Delete document failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to delete document", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 md:mx-0 md:px-6 xl:mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Document Management</h1>
          <p className="text-sm text-muted-foreground">Manage documents, metadata, and ingestion status.</p>
        </div>
        <Button className="bg-primary text-primary-foreground" onClick={handleUpload}>
          <Upload className="mr-2 h-4 w-4" />
          Upload Document
        </Button>
      </div>

      <section className="rounded-xl border bg-background shadow-sm">
        <div className="flex flex-col gap-4 border-b border-border p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold">Documents</h2>
            <p className="text-xs text-muted-foreground">
              {documents.length} total document{documents.length !== 1 ? "s" : ""}
            </p>
          </div>
          <div className="flex w-full flex-col gap-2 md:w-auto">
            <span className="text-sm font-medium text-muted-foreground">Search</span>
            <input
              type="text"
              placeholder="Search by filename..."
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary md:w-72"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th
                  className="cursor-pointer whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted/50"
                  onClick={() => handleSort("filename")}
                >
                  Filename
                  {sortField === "filename" && (sortDirection === "asc" ? " ↑" : " ↓")}
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Status
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Size
                </th>
                <th
                  className="cursor-pointer whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted/50"
                  onClick={() => handleSort("page_count")}
                >
                  Pages
                  {sortField === "page_count" && (sortDirection === "asc" ? " ↑" : " ↓")}
                </th>
                <th
                  className="cursor-pointer whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted/50"
                  onClick={() => handleSort("chunk_count")}
                >
                  Chunks
                  {sortField === "chunk_count" && (sortDirection === "asc" ? " ↑" : " ↓")}
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loadingTable ? (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-muted-foreground">
                    Loading documents...
                  </td>
                </tr>
              ) : error ? (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-destructive">
                    {error}
                  </td>
                </tr>
              ) : sortedDocuments.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-muted-foreground">
                    {searchTerm ? "No documents match your search." : "No documents found. Upload a document to get started."}
                  </td>
                </tr>
              ) : (
                sortedDocuments.map((doc) => (
                  <tr key={doc.filename} className="group transition-colors hover:bg-muted/40">
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-medium">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="max-w-xs truncate" title={doc.filename}>
                          {doc.filename}
                        </span>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <button
                        onClick={() => handleToggleActive(doc)}
                        disabled={actionSubmitting}
                        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium transition-colors ${
                          doc.is_active
                            ? "border-green-500/30 bg-green-500/10 text-green-700 dark:text-green-400"
                            : "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400"
                        }`}
                        title={doc.is_active ? "Click to disable" : "Click to enable"}
                      >
                        {doc.is_active ? (
                          <>
                            <Check className="h-3 w-3" />
                            Active
                          </>
                        ) : (
                          <>
                            <X className="h-3 w-3" />
                            Inactive
                          </>
                        )}
                      </button>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                      {formatFileSize(doc.size_bytes)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">{doc.page_count}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">{doc.chunk_count}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm">
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleViewDocument(doc)}
                          className="border-border text-xs"
                          title="View document"
                        >
                          <ExternalLink className="mr-1 h-3 w-3" />
                          View
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleEdit(doc)}
                          className="border-border text-xs"
                        >
                          <Edit className="mr-1 h-3 w-3" />
                          Edit
                        </Button>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => handleDelete(doc)}
                          className="text-xs"
                        >
                          <Trash2 className="mr-1 h-3 w-3" />
                          Delete
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Upload Modal */}
      {isUploadModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-background p-6 shadow-lg">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Upload Document</h2>
              <button
                onClick={closeAllModals}
                className="text-muted-foreground hover:text-foreground"
                disabled={actionSubmitting}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">File (PDF, DOCX, MD)</label>
                <input
                  type="file"
                  accept=".pdf,.docx,.md,.markdown"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm"
                  disabled={actionSubmitting}
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  The document will be automatically ingested into the index after upload.
                </p>
              </div>
              {uploadProgress && (
                <div className="rounded-md bg-muted/50 p-3 text-sm text-muted-foreground whitespace-pre-line">
                  {uploadProgress}
                </div>
              )}
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={closeAllModals} disabled={actionSubmitting}>
                  Cancel
                </Button>
                <Button onClick={submitUpload} disabled={!uploadFile || actionSubmitting}>
                  {actionSubmitting ? "Uploading..." : "Upload"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Edit Metadata Modal */}
      {isEditModalOpen && selectedDocument && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-background p-6 shadow-lg">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Document Information</h2>
              <button
                onClick={closeAllModals}
                className="text-muted-foreground hover:text-foreground"
                disabled={actionSubmitting}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">Filename</label>
                <input
                  type="text"
                  value={selectedDocument.filename}
                  disabled
                  className="w-full rounded-md border border-border bg-muted/50 px-3 py-2 text-sm text-muted-foreground"
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={closeAllModals} disabled={actionSubmitting}>
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && selectedDocument && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-background p-6 shadow-lg">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-destructive">Delete Document</h2>
              <button
                onClick={closeAllModals}
                className="text-muted-foreground hover:text-foreground"
                disabled={actionSubmitting}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Are you sure you want to delete <strong>{selectedDocument.filename}</strong>? This action cannot be
                undone and will remove the document from the index.
              </p>
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Type <strong>DELETE</strong> to confirm:
                </label>
                <input
                  type="text"
                  value={deleteConfirmation}
                  onChange={(e) => setDeleteConfirmation(e.target.value)}
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                  disabled={actionSubmitting}
                />
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={closeAllModals} disabled={actionSubmitting}>
                  Cancel
                </Button>
                <Button variant="destructive" onClick={submitDelete} disabled={actionSubmitting || deleteConfirmation !== "DELETE"}>
                  {actionSubmitting ? "Deleting..." : "Delete"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* View Document Modal */}
      {isViewModalOpen && selectedDocument && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="flex h-[90vh] w-[90vw] max-w-6xl flex-col rounded-lg border border-border bg-background shadow-lg">
            <div className="flex items-center justify-between border-b border-border p-4">
              <h2 className="text-xl font-semibold">{selectedDocument.filename}</h2>
              <button
                onClick={closeAllModals}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              {selectedDocument.filename.toLowerCase().endsWith('.pdf') ? (
                <iframe
                  src={`${apiBaseUrl}/documents/${encodeURIComponent(selectedDocument.filename)}`}
                  className="h-full w-full border-0"
                  title={selectedDocument.filename}
                />
              ) : selectedDocument.filename.toLowerCase().endsWith('.md') || selectedDocument.filename.toLowerCase().endsWith('.markdown') ? (
                <div className="h-full overflow-auto p-4">
                  <iframe
                    src={`${apiBaseUrl}/documents/${encodeURIComponent(selectedDocument.filename)}`}
                    className="h-full w-full border-0"
                    title={selectedDocument.filename}
                  />
                </div>
              ) : (
                <div className="flex h-full items-center justify-center p-8">
                  <div className="text-center">
                    <FileText className="mx-auto h-16 w-16 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground mb-4">
                      Preview not available for this file type. Click the link below to download.
                    </p>
                    <Button
                      variant="outline"
                      onClick={() => {
                        const encodedFilename = encodeURIComponent(selectedDocument.filename);
                        window.open(`${apiBaseUrl}/documents/${encodedFilename}`, "_blank");
                      }}
                    >
                      <ExternalLink className="mr-2 h-4 w-4" />
                      Open Document
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toastMessage && (
        <div
          className={`fixed bottom-4 right-4 z-50 rounded-lg border px-4 py-3 shadow-lg ${
            toastType === "success"
              ? "border-green-500/30 bg-green-500/10 text-green-700 dark:text-green-400"
              : "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400"
          }`}
        >
          {toastMessage}
        </div>
      )}
    </div>
  );
}

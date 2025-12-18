"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { resolveApiBaseUrl, ALLOW_APP_INGESTION } from "@/config/api";
import { Upload, FileText, Trash2, Edit, Eye, EyeOff, X, Check, ExternalLink, RefreshCw, AlertTriangle, Database, Cloud, Wrench } from "lucide-react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface Document {
  filename: string;
  size_bytes: number | null;
  uploaded_date?: string | null;
  chunk_count: number;
  page_count: number;
  file_path: string | null;
  gcs_path?: string | null;
  file_type: string | null;
  is_active: boolean;
  machine_model?: string | null | string[];
  missing_machine_model?: boolean;
  requires_admin_review?: boolean;
  category?: string | null;
  product_family?: string | null;
  ingestion_status?: string | null;
  ingestion_metadata_id?: string | null;
  ingestion_error?: string | null;
  metadata_id?: string;
  document_id?: number | null;
  display_name?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

type SortField = keyof Pick<Document, "filename" | "page_count" | "is_active">;
type SortDirection = "asc" | "desc";

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

// Helper function to get status label
const getStatusLabel = (status: string | null | undefined, chunkCount?: number): string => {
  // When ingestion is disabled, always show "Managed externally" regardless of status
  // This prevents showing misleading "Rebuilding index..." messages
  if (!ALLOW_APP_INGESTION) {
    return 'Managed externally';
  }
  
  // When ingestion is enabled, show actual status
  if (!status) {
    return chunkCount !== undefined && chunkCount > 0 ? 'Complete' : '';
  }
  
  switch (status) {
    case 'PENDING_INGESTION':
      return 'Pending';
    case 'CHUNKING':
      return 'Processing (chunking)';
    case 'READY_FOR_EMBEDDING':
      return 'Ready for embeddings';
    case 'EMBEDDING':
      return 'Processing (embedding)';
    case 'COMPLETE':
      return 'Complete';
    case 'DELETING':
      return 'Deleting…';
    case 'REBUILDING_INDEX':
      return 'Rebuilding index…';
    case 'FAILED':
      return 'Failed';
    default:
      return status;
  }
};

export default function AdminDocumentsPage() {
  // Log ingestion configuration for debugging
  console.log('[Admin Documents] ALLOW_APP_INGESTION:', ALLOW_APP_INGESTION);
  
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
  
  // Edit form state
  const [editMachineModel, setEditMachineModel] = useState<string[]>([]);  // Changed to array
  const [editCategory, setEditCategory] = useState("");
  const [editProductFamily, setEditProductFamily] = useState("");
  const [editIsActive, setEditIsActive] = useState(true);
  
  // Allowed machine models for dropdown
  const [allowedMachineModels, setAllowedMachineModels] = useState<string[]>([]);

  // Maintenance/Diagnostics state
  const [diagnosticsResult, setDiagnosticsResult] = useState<any>(null);
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [diagnosticsError, setDiagnosticsError] = useState<string | null>(null);
  const [diagnosticsLastRun, setDiagnosticsLastRun] = useState<Date | null>(null);
  const [isDiagnosticsModalOpen, setIsDiagnosticsModalOpen] = useState(false);

  const [orphansList, setOrphansList] = useState<any[]>([]);
  const [orphansLoading, setOrphansLoading] = useState(false);
  const [orphansError, setOrphansError] = useState<string | null>(null);
  const [isOrphansModalOpen, setIsOrphansModalOpen] = useState(false);

  const [deleteOrphansLoading, setDeleteOrphansLoading] = useState(false);
  const [deleteOrphansResult, setDeleteOrphansResult] = useState<any>(null);
  const [isDeleteOrphansConfirmOpen, setIsDeleteOrphansConfirmOpen] = useState(false);
  const [isDeleteOrphansFailuresOpen, setIsDeleteOrphansFailuresOpen] = useState(false);

  const apiBaseUrl = useMemo(() => resolveApiBaseUrl(), []);

  const showToast = useCallback((message: string, type: "success" | "error" = "success") => {
    setToastType(type);
    setToastMessage(message);
    window.setTimeout(() => setToastMessage(null), 3000);
  }, []);

  // Robust response parsing: avoid blindly calling response.json() on HTML error pages.
  // Always read text once; parse JSON only when content-type indicates JSON.
  const readResponseBody = useCallback(async (response: Response) => {
    const contentType = response.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");

    const rawText = await response.text().catch(() => "");
    const preview = rawText.slice(0, 300);

    if (!isJson) {
      return { contentType, isJson: false as const, text: preview };
    }

    try {
      const json = rawText ? JSON.parse(rawText) : {};
      return { contentType, isJson: true as const, json };
    } catch {
      return { contentType, isJson: false as const, text: preview };
    }
  }, []);

  const [roleForPage, setRoleForPage] = useState<string | null>(null);

  const extractApiError = useCallback(function extractApiErrorInner(detail: unknown): string | null {
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
        return extractApiErrorInner(nested);
      }
      try {
        return JSON.stringify(detail);
      } catch (error) {
        return String(detail);
      }
    }
    return String(detail);
  }, []);

  const requireAdminOrExplain = useCallback(async (): Promise<boolean> => {
    // Extra safety: admin pages should be gated by layout, but don't assume.
    try {
      const meResp = await fetch(`/api/auth/me`, { credentials: "include" });
      const body = await readResponseBody(meResp);
      if (!meResp.ok) {
        const msg =
          body.isJson ? extractApiError(body.json) : `Not authenticated (${meResp.status}): ${body.text || body.contentType}`;
        showToast(msg || "Not authenticated", "error");
        return false;
      }
      const role = body.isJson ? (body.json as any)?.role : null;
      if (role !== "ADMIN") {
        showToast("Admin only: Diagnostics are restricted to ADMIN users.", "error");
        return false;
      }
      return true;
    } catch (err) {
      showToast("Admin only: Unable to verify permissions.", "error");
      return false;
    }
  }, [extractApiError, readResponseBody, showToast]);

  const fetchAllowedMachineModels = useCallback(
    async () => {
      try {
        // Fetch machine models from DB table (via /admin/machines endpoint)
        // This ensures we only show machine models that exist in the database
        // Cookie-based JWT is automatically sent with fetch requests
        const response = await fetch(`${apiBaseUrl}/admin/machines`, {
          credentials: "include",
        });
        if (!response.ok) {
          const body = await readResponseBody(response);
          console.warn(`Failed to fetch machine models: ${response.status}`, body);
          return;
        }
        const body = await readResponseBody(response);
        if (!body.isJson) {
          throw new Error(`Failed to parse machine models response (${response.status}): ${body.text || body.contentType}`);
        }
        const data = body.json;
        // Extract machine names from the machines array
        // Handle both array format and object with machines property
        let machines: Array<{ name: string }> = [];
        if (Array.isArray(data)) {
          machines = data;
        } else if (data.machines && Array.isArray(data.machines)) {
          machines = data.machines;
        }
        const machineNames = machines.map((m: { name: string }) => m.name).filter(Boolean);
        setAllowedMachineModels(machineNames);
      } catch (err) {
        console.warn("Failed to fetch machine models:", err);
        // Fallback: try to get from documents response
      }
    },
    [apiBaseUrl, readResponseBody]
  );

  const fetchDocuments = useCallback(
    async () => {
      setLoadingTable(true);
      setError(null);
      try {
        // Cookie-based JWT is automatically sent with fetch requests
        const response = await fetch(`${apiBaseUrl}/admin/documents`, {
          credentials: "include",
        });
        if (!response.ok) {
          const body = await readResponseBody(response);
          const detail = body.isJson ? extractApiError(body.json) : body.text;
          throw new Error(detail || `Failed to load documents (${response.status})`);
        }
        const body = await readResponseBody(response);
        if (!body.isJson) {
          throw new Error(`Documents API returned non-JSON (${response.status}): ${(body.text || body.contentType).slice(0, 200)}`);
        }
        const data = body.json;
        const docs = Array.isArray(data.documents) ? data.documents : [];
        setDocuments(docs);
        
        // Log sample document status for debugging
        if (docs.length > 0 && !ALLOW_APP_INGESTION) {
          const sampleDoc = docs[0];
          console.log('[Admin Documents] Sample document status:', {
            filename: sampleDoc.filename,
            ingestion_status: sampleDoc.ingestion_status,
            chunk_count: sampleDoc.chunk_count,
          });
        }
        
        // Fallback: if machines endpoint failed, try to get from documents response
        // This is a fallback only - primary source should be /admin/machines
      } catch (err) {
        console.error("Failed to fetch documents:", err);
        setError(err instanceof Error ? err.message : "Unable to load documents.");
      } finally {
        setLoadingTable(false);
      }
    },
    [apiBaseUrl, extractApiError, readResponseBody]
  );

  useEffect(() => {
    // Cookie-based JWT is automatically sent with fetch requests
    // Admin layout already verified user is ADMIN
    fetchDocuments();
    fetchAllowedMachineModels();
  }, [fetchDocuments, fetchAllowedMachineModels]);

  // Defensive: determine role for UI gating (prevents CUSTOMER from seeing/calling admin diagnostics if routing misbehaves).
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const resp = await fetch(`/api/auth/me`, { credentials: "include" });
        const body = await readResponseBody(resp);
        if (!mounted) return;
        if (!resp.ok || !body.isJson) {
          setRoleForPage(null);
          return;
        }
        const role = (body.json as any)?.role;
        setRoleForPage(typeof role === "string" ? role : null);
      } catch {
        if (!mounted) return;
        setRoleForPage(null);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [readResponseBody]);

  // Poll for documents with active ingestion status (only when ingestion is enabled and page is visible)
  useEffect(() => {
    // When ingestion is disabled, don't poll at all
    if (!ALLOW_APP_INGESTION) {
      return;
    }
    
    const activeStatuses = ['PENDING_INGESTION', 'CHUNKING', 'READY_FOR_EMBEDDING', 'EMBEDDING', 'DELETING', 'REBUILDING_INDEX'];
    
    // Check if there are any active ingestions
    const checkActiveIngestions = () => {
      return documents.some(
        doc => doc.ingestion_status && activeStatuses.includes(doc.ingestion_status)
      );
    };

    const hasActiveIngestion = checkActiveIngestions();

    if (!hasActiveIngestion) {
      // No active ingestion, don't poll
      return;
    }

    // Poll every 5 seconds when ingestion is enabled and there are active ingestions
    const interval = setInterval(() => {
      // Only poll if page is visible
      if (document.hidden) {
        return;
      }
      
      // Fetch documents - the effect will re-run and check if polling should continue
      fetchDocuments();
    }, 5000);

    return () => clearInterval(interval);
  }, [documents, fetchDocuments]);

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
    setEditMachineModel([]);
    setEditCategory("");
    setEditProductFamily("");
    setEditIsActive(true);
  };

  const closeAllModals = () => {
    setIsUploadModalOpen(false);
    setIsEditModalOpen(false);
    setIsDeleteModalOpen(false);
    setIsViewModalOpen(false);
    setIsDiagnosticsModalOpen(false);
    setIsOrphansModalOpen(false);
    setIsDeleteOrphansConfirmOpen(false);
    setIsDeleteOrphansFailuresOpen(false);
    resetFormState();
  };

  const handleUpload = () => {
    resetFormState();
    setIsUploadModalOpen(true);
    // Refresh machine models list when opening upload modal to ensure latest models are available
    fetchAllowedMachineModels();
  };

  const handleEdit = (doc: Document) => {
    setSelectedDocument(doc);
    // Normalize machine_model to array (handle both string and array formats)
    const machineModels = doc.machine_model 
      ? (Array.isArray(doc.machine_model) ? doc.machine_model : [doc.machine_model])
      : [];
    setEditMachineModel(machineModels);
    setEditCategory(doc.category || "");
    setEditProductFamily(doc.product_family || "");
    setEditIsActive(doc.is_active);
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
    setActionSubmitting(true);
    try {
      const encodedFilename = encodeURIComponent(doc.filename);
      // Cookie-based JWT is automatically sent with fetch requests
      const response = await fetch(`${apiBaseUrl}/admin/documents/${encodedFilename}/toggle`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({ is_active: !doc.is_active }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to toggle document status");
      }
      showToast(`✅ Document ${!doc.is_active ? "enabled" : "disabled"}`);
      await fetchDocuments();
    } catch (err) {
      console.error("Toggle document status failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to toggle document status", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  const submitUpload = async () => {
    if (!uploadFile) return;
    
    // Validate machine model is selected
    if (!editMachineModel || editMachineModel.length === 0) {
      showToast("Please select at least one machine model", "error");
      return;
    }
    
    setActionSubmitting(true);
    setUploadProgress("Uploading file...");
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      
      // Append machine model (use first one if array, or the string value)
      const machineModelValue = Array.isArray(editMachineModel) 
        ? editMachineModel[0] 
        : editMachineModel;
      formData.append("machine_model", machineModelValue);
      
      // Append description if provided
      if (editProductFamily && editProductFamily.trim()) {
        formData.append("description", editProductFamily.trim());
      }

      setUploadProgress("Uploading file to server...");
      // Cookie-based JWT is automatically sent with fetch requests
      const response = await fetch(`${apiBaseUrl}/admin/documents/upload`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });

      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(extractApiError(detail) || "Failed to upload document");
      }

      if (ALLOW_APP_INGESTION) {
        setUploadProgress("Ingesting document into index (this may take a moment)...");
      } else {
        setUploadProgress("Saving document metadata...");
      }
      const result = await response.json();
      
      if (ALLOW_APP_INGESTION) {
        setUploadProgress(
          `✅ Complete! Processed ${result.page_count || 0} pages. Reloading index...`
        );
        showToast(`✅ Document uploaded and ingested successfully`);
      } else {
        setUploadProgress(`✅ Document uploaded. Metadata saved. Ingestion must be triggered via external GPU pipeline.`);
        showToast(`✅ Document uploaded. Ingestion will be handled externally.`);
      }
      await fetchDocuments();
      
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


  const submitEdit = async () => {
    if (!selectedDocument) return;
    setActionSubmitting(true);
    try {
      const encodedFilename = encodeURIComponent(selectedDocument.filename);
      
      // Compare machine models (handle both array and string formats)
      const currentMachineModels = selectedDocument.machine_model 
        ? (Array.isArray(selectedDocument.machine_model) 
            ? selectedDocument.machine_model 
            : [selectedDocument.machine_model])
        : [];
      const modelsEqual = editMachineModel.length === currentMachineModels.length &&
        editMachineModel.every((model, idx) => model === currentMachineModels[idx]);
      
      // Build update body
      const body: Record<string, unknown> = {};
      if (!modelsEqual) {
        body.machine_model = editMachineModel.length > 0 ? editMachineModel : null;
      }
      if (editCategory !== (selectedDocument.category || "")) {
        body.category = editCategory || null;
      }
      if (editProductFamily !== (selectedDocument.product_family || "")) {
        body.product_family = editProductFamily || null;
      }
      if (editIsActive !== selectedDocument.is_active) {
        body.is_active = editIsActive;
      }
      
      // Update via metadata endpoint if any changes
      if (Object.keys(body).length > 0) {
        // Cookie-based JWT is automatically sent with fetch requests
        const response = await fetch(`${apiBaseUrl}/admin/documents/${encodedFilename}/metadata`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          credentials: "include",
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          const detail = await response.json().catch(() => null);
          throw new Error(extractApiError(detail) || "Failed to update document");
        }
      }
      
      showToast("✅ Document metadata updated");
      await fetchDocuments();
      closeAllModals();
    } catch (err) {
      console.error("Edit document failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to update document", "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  const submitDelete = async () => {
    if (!selectedDocument) return;
    if (deleteConfirmation !== "DELETE") {
      showToast("Please type DELETE to confirm", "error");
      return;
    }
    setActionSubmitting(true);
    try {
      // Use metadata_id endpoint (preferred) or ingestion_metadata_id (fallback)
      const metadataId = selectedDocument.metadata_id || selectedDocument.ingestion_metadata_id;
      if (!metadataId) {
        throw new Error("Document metadata ID not found. Cannot delete.");
      }
      
      // Use the reliable delete endpoint that works even if GCS object is missing
      // Go through Next.js API route for proper authentication
      const response = await fetch(`/api/admin/documents/metadata/${metadataId}`, {
          method: "DELETE",
          credentials: "include",
        });
      
      if (!response.ok) {
        let errorDetail = null;
        try {
          errorDetail = await response.json();
        } catch {
          // Response might not be JSON
          errorDetail = { detail: `HTTP ${response.status}: ${response.statusText}` };
        }
        const errorMessage = extractApiError(errorDetail) || `Failed to delete document (${response.status})`;
        throw new Error(errorMessage);
      }
      
      // Close modal immediately and reset state
      closeAllModals();
      setDeleteConfirmation("");
      
      // Show success message - deletion always succeeds, index cleanup is best-effort
        showToast("✅ Document deleted successfully.", "success");
      await fetchDocuments();
    } catch (err) {
      console.error("Delete document failed:", err);
      // Extract detailed error message from response
      let errorMessage = "Failed to delete document";
      if (err instanceof Error) {
        errorMessage = err.message;
      }
      // Show the actual error message from backend
      showToast(errorMessage, "error");
    } finally {
      setActionSubmitting(false);
    }
  };

  // Maintenance functions
  const runDiagnostics = useCallback(async () => {
    setDiagnosticsLoading(true);
    setDiagnosticsError(null);
    try {
      const isAdmin = await requireAdminOrExplain();
      if (!isAdmin) return;

      const response = await fetch(`/api/admin/documents/diagnostics`, {
        credentials: "include",
      });
      if (!response.ok) {
        const body = await readResponseBody(response);
        const detail = body.isJson ? extractApiError(body.json) : body.text;
        throw new Error(detail || `Diagnostics failed (${response.status}): ${(body.text || body.contentType).slice(0, 200)}`);
      }
      const body = await readResponseBody(response);
      if (!body.isJson) {
        throw new Error(
          `Diagnostics returned non-JSON (${response.status}). ${body.text ? body.text.slice(0, 200) : body.contentType}`
        );
      }
      const data = body.json;
      setDiagnosticsResult(data);
      setDiagnosticsLastRun(new Date());
      setIsDiagnosticsModalOpen(true);
    } catch (err) {
      console.error("Diagnostics failed:", err);
      setDiagnosticsError(err instanceof Error ? err.message : "Failed to get diagnostics");
      showToast(err instanceof Error ? err.message : "Failed to get diagnostics", "error");
    } finally {
      setDiagnosticsLoading(false);
    }
  }, [extractApiError, readResponseBody, requireAdminOrExplain, showToast]);

  const viewOrphans = useCallback(async () => {
    setOrphansLoading(true);
    setOrphansError(null);
    try {
      const isAdmin = await requireAdminOrExplain();
      if (!isAdmin) return;

      const response = await fetch(`/api/admin/documents/orphans`, {
        credentials: "include",
      });
      if (!response.ok) {
        const body = await readResponseBody(response);
        const detail = body.isJson ? extractApiError(body.json) : body.text;
        throw new Error(detail || `Failed to get orphans (${response.status})`);
      }
      const body = await readResponseBody(response);
      if (!body.isJson) {
        throw new Error(`Orphans returned non-JSON (${response.status}): ${body.text || body.contentType}`);
      }
      const data = body.json;
      setOrphansList(data.orphans || []);
      setIsOrphansModalOpen(true);
    } catch (err) {
      console.error("View orphans failed:", err);
      setOrphansError(err instanceof Error ? err.message : "Failed to get orphans");
      showToast(err instanceof Error ? err.message : "Failed to get orphans", "error");
    } finally {
      setOrphansLoading(false);
    }
  }, [extractApiError, readResponseBody, requireAdminOrExplain, showToast]);

  const deleteAllOrphans = useCallback(async () => {
    setDeleteOrphansLoading(true);
    try {
      const isAdmin = await requireAdminOrExplain();
      if (!isAdmin) return;

      const response = await fetch(`/api/admin/documents/orphans`, {
        method: "DELETE",
        credentials: "include",
      });
      if (!response.ok) {
        const body = await readResponseBody(response);
        const detail = body.isJson ? extractApiError(body.json) : body.text;
        throw new Error(detail || `Failed to delete orphans (${response.status})`);
      }
      const body = await readResponseBody(response);
      if (!body.isJson) {
        throw new Error(`Delete orphans returned non-JSON (${response.status}): ${body.text || body.contentType}`);
      }
      const data = body.json;
      setDeleteOrphansResult(data);
      
      // Show success message
      const successMsg = `✅ Deleted ${data.count_deleted || 0} orphan record(s)`;
      if (data.count_failed > 0) {
        showToast(`${successMsg}. ${data.count_failed} failed.`, "error");
        setIsDeleteOrphansFailuresOpen(true);
      } else {
        showToast(successMsg, "success");
      }
      
      // Close confirmation modal
      setIsDeleteOrphansConfirmOpen(false);
      
      // Refresh documents list
      await fetchDocuments();
      
      // Refresh orphans list if modal is open
      if (isOrphansModalOpen) {
        await viewOrphans();
      }
    } catch (err) {
      console.error("Delete orphans failed:", err);
      showToast(err instanceof Error ? err.message : "Failed to delete orphans", "error");
    } finally {
      setDeleteOrphansLoading(false);
    }
  }, [extractApiError, fetchDocuments, isOrphansModalOpen, readResponseBody, requireAdminOrExplain, showToast, viewOrphans]);

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

      {/* Maintenance Section */}
      <section className="rounded-xl border bg-background shadow-sm p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Wrench className="h-5 w-5 text-muted-foreground" />
            <h2 className="text-lg font-semibold">Maintenance</h2>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {roleForPage !== null && roleForPage !== "ADMIN" ? (
              <div className="text-sm text-muted-foreground">
                Admin only: Diagnostics and maintenance tools are disabled for your account.
              </div>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={runDiagnostics}
                  disabled={diagnosticsLoading}
                >
                  {diagnosticsLoading ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Running...
                    </>
                  ) : (
                    <>
                      <Database className="mr-2 h-4 w-4" />
                      Diagnostics
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={viewOrphans}
                  disabled={orphansLoading}
                >
                  {orphansLoading ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="mr-2 h-4 w-4" />
                      View Orphans
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    // Load orphans first to show count in confirmation
                    if (orphansList.length === 0) {
                      await viewOrphans();
                    }
                    setIsDeleteOrphansConfirmOpen(true);
                  }}
                  disabled={deleteOrphansLoading || orphansLoading}
                  className="text-destructive hover:text-destructive"
                >
                  {deleteOrphansLoading ? (
                    <>
                      <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                      Deleting...
                    </>
                  ) : (
                    <>
                      <Trash2 className="mr-2 h-4 w-4" />
                      Delete All Orphans
                    </>
                  )}
                </Button>
              </>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={fetchDocuments}
              disabled={loadingTable}
            >
              {loadingTable ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Refreshing...
                </>
              ) : (
                <>
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </>
              )}
            </Button>
          </div>
        </div>
        {diagnosticsLastRun && (
          <p className="mt-2 text-xs text-muted-foreground">
            Last diagnostics run: {diagnosticsLastRun.toLocaleString()}
          </p>
        )}
      </section>

      <section className="rounded-xl border bg-background shadow-sm">
        <div className="flex flex-col gap-4 border-b border-border p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold">Documents</h2>
            <p className="text-xs text-muted-foreground">
              {documents.length} total document{documents.length !== 1 ? "s" : ""}
              {documents.filter(d => d.missing_machine_model || d.requires_admin_review).length > 0 && (
                <span className="ml-2 text-orange-600 dark:text-orange-400">
                  • {documents.filter(d => d.missing_machine_model || d.requires_admin_review).length} need review
                </span>
              )}
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
                  Machine Model
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
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Ingestion Status
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loadingTable ? (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-muted-foreground">
                    Loading documents...
                  </td>
                </tr>
              ) : error ? (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-destructive">
                    {error}
                  </td>
                </tr>
              ) : sortedDocuments.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-muted-foreground">
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
                    <td className="whitespace-nowrap px-4 py-3 text-sm">
                      {doc.missing_machine_model || !doc.machine_model || (Array.isArray(doc.machine_model) && doc.machine_model.length === 0) ? (
                        <div className="flex items-center gap-2">
                          <span className="text-red-600 dark:text-red-400 font-medium">Missing</span>
                          {doc.requires_admin_review && (
                            <span className="inline-flex items-center gap-1 rounded-full border border-orange-500/30 bg-orange-500/10 px-2 py-0.5 text-xs font-medium text-orange-700 dark:text-orange-400">
                              Needs review
                            </span>
                          )}
                        </div>
                      ) : (
                        <span className="text-muted-foreground">
                          {Array.isArray(doc.machine_model) 
                            ? doc.machine_model.join(", ")
                            : doc.machine_model}
                        </span>
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                      {formatFileSize(doc.size_bytes || 0)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">{doc.page_count}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm">
                      {(() => {
                        const statusLabel = getStatusLabel(doc.ingestion_status, doc.chunk_count);
                        if (!statusLabel) {
                          return <span className="text-muted-foreground text-xs">—</span>;
                        }
                        
                        // When ingestion is disabled, always use neutral outline style
                        const badgeClass = !ALLOW_APP_INGESTION
                          ? 'border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400'
                          : doc.ingestion_status === 'FAILED'
                          ? 'border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400'
                          : doc.ingestion_status === 'PENDING_INGESTION'
                          ? 'border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400'
                          : doc.ingestion_status === 'COMPLETE'
                          ? 'border-green-500/30 bg-green-500/10 text-green-700 dark:text-green-400'
                          : 'border-blue-500/30 bg-blue-500/10 text-blue-700 dark:text-blue-400';
                        
                        return (
                          <div className="flex flex-col gap-1">
                            <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${badgeClass}`}>
                              {statusLabel}
                            </span>
                            {doc.ingestion_error && ALLOW_APP_INGESTION && (
                              <span className="text-xs text-red-600 dark:text-red-400 truncate max-w-xs" title={doc.ingestion_error}>
                                {doc.ingestion_error.substring(0, 40)}...
                              </span>
                            )}
                          </div>
                        );
                      })()}
                    </td>
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
                <label className="mb-2 block text-sm font-medium">File (PDF, DOCX, MD) *</label>
                <input
                  type="file"
                  accept=".pdf,.docx,.md,.markdown"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm"
                  disabled={actionSubmitting}
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  {ALLOW_APP_INGESTION 
                    ? "The document will be automatically ingested into the index after upload."
                    : "The document will be saved for metadata. Ingestion must be triggered via external GPU pipeline."}
                </p>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Machine Model *</label>
                <select
                  value={Array.isArray(editMachineModel) && editMachineModel.length > 0 ? editMachineModel[0] : ""}
                  onChange={(e) => setEditMachineModel([e.target.value])}
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                  disabled={actionSubmitting || allowedMachineModels.length === 0}
                  required
                >
                  <option value="">Select a machine model</option>
                  {allowedMachineModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
                {allowedMachineModels.length === 0 && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    No machine models available. Please add one first.
                  </p>
                )}
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Description (Optional)</label>
                <textarea
                  value={editProductFamily || ""}
                  onChange={(e) => setEditProductFamily(e.target.value)}
                  placeholder="Optional description for this document"
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm min-h-[80px] outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                  disabled={actionSubmitting}
                  rows={3}
                />
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
                <Button 
                  onClick={submitUpload} 
                  disabled={!uploadFile || !editMachineModel || (Array.isArray(editMachineModel) && editMachineModel.length === 0) || actionSubmitting}
                >
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
          <div className="w-full max-w-md rounded-lg border border-border bg-background p-6 shadow-lg max-h-[90vh] overflow-y-auto">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Edit Document Metadata</h2>
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
              <div>
                <label className="mb-2 block text-sm font-medium">
                  Machine Model(s) <span className="text-red-500">*</span>
                </label>
                <div className="space-y-2">
                  <select
                    multiple
                    value={editMachineModel}
                    onChange={(e) => {
                      const selected = Array.from(e.target.selectedOptions, option => option.value);
                      setEditMachineModel(selected);
                    }}
                    size={Math.min(allowedMachineModels.length + 1, 8)}
                    className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                    disabled={actionSubmitting}
                  >
                    {allowedMachineModels.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </select>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Select one or more machine models. Hold Ctrl/Cmd to select multiple. 
                    Select &quot;GENERAL&quot; if this document applies to all users regardless of machine.
                    {editMachineModel.length > 0 && ` (${editMachineModel.length} selected)`}
                  </p>
                  {selectedDocument.requires_admin_review && (
                    <p className="mt-1 text-xs text-orange-600 dark:text-orange-400">
                      ⚠️ This document requires admin review (machine model was not automatically detected).
                    </p>
                  )}
                </div>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Category</label>
                <input
                  type="text"
                  value={editCategory}
                  onChange={(e) => setEditCategory(e.target.value)}
                  placeholder="Optional category"
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                  disabled={actionSubmitting}
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Product Family</label>
                <input
                  type="text"
                  value={editProductFamily}
                  onChange={(e) => setEditProductFamily(e.target.value)}
                  placeholder="Optional product family"
                  className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                  disabled={actionSubmitting}
                />
              </div>
              <div>
                <label className="mb-2 flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={editIsActive}
                    onChange={(e) => setEditIsActive(e.target.checked)}
                    className="rounded border-border"
                    disabled={actionSubmitting}
                  />
                  <span className="text-sm font-medium">Document is active</span>
                </label>
                <p className="mt-1 text-xs text-muted-foreground">
                  Inactive documents will be excluded from search results.
                </p>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={closeAllModals} disabled={actionSubmitting}>
                  Cancel
                </Button>
                <Button onClick={submitEdit} disabled={actionSubmitting}>
                  {actionSubmitting ? "Saving..." : "Save Changes"}
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

      {/* Diagnostics Modal */}
      {isDiagnosticsModalOpen && diagnosticsResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-2xl rounded-lg border border-border bg-background p-6 shadow-lg max-h-[90vh] overflow-y-auto">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Document Diagnostics</h2>
              <button
                onClick={() => setIsDiagnosticsModalOpen(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="rounded-md border border-border bg-muted/30 p-3">
                  <div className="text-xs font-medium text-muted-foreground">DB Metadata Records</div>
                  <div className="text-2xl font-semibold">{diagnosticsResult.count_db_metadata || 0}</div>
                </div>
                <div className="rounded-md border border-border bg-muted/30 p-3">
                  <div className="text-xs font-medium text-muted-foreground">DB Document Records</div>
                  <div className="text-2xl font-semibold">{diagnosticsResult.count_db_documents || 0}</div>
                </div>
                <div className="rounded-md border border-border bg-muted/30 p-3">
                  <div className="text-xs font-medium text-muted-foreground">GCS Objects</div>
                  <div className="text-2xl font-semibold">{diagnosticsResult.count_gcs_objects || 0}</div>
                </div>
                <div className="rounded-md border border-border bg-muted/30 p-3">
                  <div className="text-xs font-medium text-muted-foreground">Orphaned Metadata</div>
                  <div className="text-2xl font-semibold text-orange-600 dark:text-orange-400">
                    {diagnosticsResult.orphan_metadata_ids?.length || 0}
                  </div>
                </div>
              </div>

              {diagnosticsResult.orphan_metadata_ids && diagnosticsResult.orphan_metadata_ids.length > 0 && (
                <div>
                  <h3 className="mb-2 text-sm font-semibold text-orange-600 dark:text-orange-400">
                    Orphaned Metadata Records ({diagnosticsResult.orphan_metadata_ids.length})
                  </h3>
                  <div className="max-h-48 overflow-y-auto rounded-md border border-border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="text-xs">Metadata ID</TableHead>
                          <TableHead className="text-xs">Filename</TableHead>
                          <TableHead className="text-xs">GCS Path</TableHead>
                          <TableHead className="text-xs">Reason</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {diagnosticsResult.orphan_metadata_ids.map((orphan: any, idx: number) => (
                          <TableRow key={idx}>
                            <TableCell className="text-xs font-mono">{orphan.metadata_id?.substring(0, 8)}...</TableCell>
                            <TableCell className="text-xs">{orphan.filename}</TableCell>
                            <TableCell className="text-xs font-mono text-muted-foreground truncate max-w-xs" title={orphan.gcs_path}>
                              {orphan.gcs_path}
                            </TableCell>
                            <TableCell className="text-xs text-orange-600 dark:text-orange-400">{orphan.reason}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}

              {diagnosticsResult.gcs_objects_without_db && diagnosticsResult.gcs_objects_without_db.length > 0 && (
                <div>
                  <h3 className="mb-2 text-sm font-semibold text-blue-600 dark:text-blue-400">
                    GCS Objects Without DB Records ({diagnosticsResult.gcs_objects_without_db.length})
                  </h3>
                  <div className="max-h-48 overflow-y-auto rounded-md border border-border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="text-xs">Object Name</TableHead>
                          <TableHead className="text-xs">GCS Path</TableHead>
                          <TableHead className="text-xs">Reason</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {diagnosticsResult.gcs_objects_without_db.map((obj: any, idx: number) => (
                          <TableRow key={idx}>
                            <TableCell className="text-xs">{obj.object_name}</TableCell>
                            <TableCell className="text-xs font-mono text-muted-foreground truncate max-w-xs" title={obj.gcs_path}>
                              {obj.gcs_path}
                            </TableCell>
                            <TableCell className="text-xs text-blue-600 dark:text-blue-400">{obj.reason}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}

              {(!diagnosticsResult.orphan_metadata_ids || diagnosticsResult.orphan_metadata_ids.length === 0) &&
               (!diagnosticsResult.gcs_objects_without_db || diagnosticsResult.gcs_objects_without_db.length === 0) && (
                <div className="rounded-md border border-green-500/30 bg-green-500/10 p-3 text-center text-sm text-green-700 dark:text-green-400">
                  ✅ All records are consistent
                </div>
              )}

              <div className="flex justify-end">
                <Button variant="outline" onClick={() => setIsDiagnosticsModalOpen(false)}>
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Orphans Modal */}
      {isOrphansModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-4xl rounded-lg border border-border bg-background p-6 shadow-lg max-h-[90vh] overflow-y-auto">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold">Orphaned Documents</h2>
              <button
                onClick={() => setIsOrphansModalOpen(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              {orphansError ? (
                <div className="rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-400">
                  {orphansError}
                </div>
              ) : orphansList.length === 0 ? (
                <div className="rounded-md border border-green-500/30 bg-green-500/10 p-3 text-center text-sm text-green-700 dark:text-green-400">
                  ✅ No orphaned records found
                </div>
              ) : (
                <>
                  <p className="text-sm text-muted-foreground">
                    Found {orphansList.length} orphaned record(s). These are database records that reference missing GCS objects or have no file path.
                  </p>
                  <div className="max-h-96 overflow-y-auto rounded-md border border-border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="text-xs">Metadata ID</TableHead>
                          <TableHead className="text-xs">Filename</TableHead>
                          <TableHead className="text-xs">GCS Path</TableHead>
                          <TableHead className="text-xs">File Path</TableHead>
                          <TableHead className="text-xs">Reason</TableHead>
                          <TableHead className="text-xs">Created At</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {orphansList.map((orphan: any, idx: number) => (
                          <TableRow key={idx}>
                            <TableCell className="text-xs font-mono">{orphan.metadata_id?.substring(0, 8)}...</TableCell>
                            <TableCell className="text-xs">{orphan.filename}</TableCell>
                            <TableCell className="text-xs font-mono text-muted-foreground truncate max-w-xs" title={orphan.gcs_path}>
                              {orphan.gcs_path || "—"}
                            </TableCell>
                            <TableCell className="text-xs font-mono text-muted-foreground truncate max-w-xs" title={orphan.file_path}>
                              {orphan.file_path || "—"}
                            </TableCell>
                            <TableCell className="text-xs text-orange-600 dark:text-orange-400">{orphan.reason}</TableCell>
                            <TableCell className="text-xs text-muted-foreground">
                              {orphan.created_at ? new Date(orphan.created_at).toLocaleDateString() : "—"}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </>
              )}
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setIsOrphansModalOpen(false)}>
                  Close
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete All Orphans Confirmation Modal */}
      {isDeleteOrphansConfirmOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-md rounded-lg border border-border bg-background p-6 shadow-lg">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-destructive">Delete All Orphans</h2>
              <button
                onClick={() => setIsDeleteOrphansConfirmOpen(false)}
                className="text-muted-foreground hover:text-foreground"
                disabled={deleteOrphansLoading}
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                This will permanently delete all orphaned document records from the database.
                {orphansList.length > 0 && (
                  <span className="block mt-2 font-semibold text-foreground">
                    {orphansList.length} orphaned record(s) will be deleted.
                  </span>
                )}
              </p>
              <p className="text-sm text-muted-foreground">
                This action cannot be undone. Orphaned records are database entries that reference missing GCS objects or have no file path.
              </p>
              <div className="flex justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={() => setIsDeleteOrphansConfirmOpen(false)}
                  disabled={deleteOrphansLoading}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  onClick={deleteAllOrphans}
                  disabled={deleteOrphansLoading}
                >
                  {deleteOrphansLoading ? "Deleting..." : "Delete All Orphans"}
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Delete Orphans Failures Modal */}
      {isDeleteOrphansFailuresOpen && deleteOrphansResult && deleteOrphansResult.failures && deleteOrphansResult.failures.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="w-full max-w-2xl rounded-lg border border-border bg-background p-6 shadow-lg max-h-[90vh] overflow-y-auto">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-destructive">Deletion Failures</h2>
              <button
                onClick={() => setIsDeleteOrphansFailuresOpen(false)}
                className="text-muted-foreground hover:text-foreground"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {deleteOrphansResult.count_deleted || 0} orphan(s) deleted successfully.
                {deleteOrphansResult.count_failed > 0 && (
                  <span className="block mt-2 font-semibold text-destructive">
                    {deleteOrphansResult.count_failed} deletion(s) failed:
                  </span>
                )}
              </p>
              <div className="max-h-96 overflow-y-auto rounded-md border border-border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Metadata ID</TableHead>
                      <TableHead className="text-xs">Failure Reason</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {deleteOrphansResult.failures.map((failure: any, idx: number) => (
                      <TableRow key={idx}>
                        <TableCell className="text-xs font-mono">{failure.metadata_id?.substring(0, 8)}...</TableCell>
                        <TableCell className="text-xs text-red-600 dark:text-red-400">{failure.reason}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              <div className="flex justify-end">
                <Button variant="outline" onClick={() => setIsDeleteOrphansFailuresOpen(false)}>
                  Close
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

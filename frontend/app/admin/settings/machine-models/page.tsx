"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Trash2, Loader2, Edit2 } from "lucide-react";

interface MachineModel {
  id: number;
  name: string;
  machine_kind: string;
  document_count: number;
  created_at: string;
}

interface MachineModelsResponse {
  machines: MachineModel[];
  total_documents: number;
  matched_documents: number;
  unmatched_documents: number;
  unmatched_machine_models: string[];
}

interface ModalProps {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

function Modal({ title, onClose, children }: ModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-xl border border-border bg-background shadow-xl">
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <h2 className="text-lg font-semibold">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
          >
            ✕
          </button>
        </div>
        <div className="px-4 py-5">{children}</div>
      </div>
    </div>
  );
}

export default function MachineModelsPage() {
  const [machines, setMachines] = useState<MachineModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalDocuments, setTotalDocuments] = useState<number>(0);
  const [matchedDocuments, setMatchedDocuments] = useState<number>(0);
  const [unmatchedDocuments, setUnmatchedDocuments] = useState<number>(0);
  const [unmatchedMachineModels, setUnmatchedMachineModels] = useState<string[]>([]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editingMachine, setEditingMachine] = useState<MachineModel | null>(null);
  const [newMachineName, setNewMachineName] = useState("");
  const [newMachineKind, setNewMachineKind] = useState("Print Engine");
  const [editingMachineName, setEditingMachineName] = useState("");
  const [editingMachineKind, setEditingMachineKind] = useState("Print Engine");
  const [submitting, setSubmitting] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [machineToDelete, setMachineToDelete] = useState<MachineModel | null>(null);
  const [deleting, setDeleting] = useState(false);
  const { toast } = useToast();

  const MACHINE_KINDS = ["Print Engine", "Blade Cutter", "Laser Cutter", "Printer"] as const;

  const fetchMachines = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // Cookie-based JWT is automatically sent with fetch requests
      const response = await fetch("/api/admin/machines", {
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to fetch machine models");
      }

      const data = await response.json().catch(() => null);
      
      // Handle null/undefined response
      if (!data) {
        setMachines([]);
        setTotalDocuments(0);
        setMatchedDocuments(0);
        setUnmatchedDocuments(0);
        setUnmatchedMachineModels([]);
        return;
      }

      // Handle both old format (array) and new format (object with machines array)
      if (Array.isArray(data)) {
        setMachines(data);
        setTotalDocuments(0);
        setMatchedDocuments(0);
        setUnmatchedDocuments(0);
        setUnmatchedMachineModels([]);
      } else if (data && typeof data === 'object' && 'machines' in data) {
        // Ensure machines is always an array
        const machinesArray = Array.isArray(data.machines) ? data.machines : [];
        setMachines(machinesArray);
        setTotalDocuments(data.total_documents || 0);
        setMatchedDocuments(data.matched_documents || 0);
        setUnmatchedDocuments(data.unmatched_documents || 0);
        setUnmatchedMachineModels(Array.isArray(data.unmatched_machine_models) ? data.unmatched_machine_models : []);
        // Log unmatched documents for debugging
        if (data.unmatched_documents > 0) {
          console.warn(
            `Found ${data.unmatched_documents} documents with unmatched machine models:`,
            data.unmatched_machine_models
          );
        }
      } else {
        // Unexpected format - default to empty array
        console.warn("Unexpected response format:", data);
        setMachines([]);
        setTotalDocuments(0);
        setMatchedDocuments(0);
        setUnmatchedDocuments(0);
        setUnmatchedMachineModels([]);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to fetch machine models";
      setError(errorMessage);
      setMachines([]); // Ensure machines is always an array even on error
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchMachines();
  }, [fetchMachines]);

  const handleAddMachine = async () => {
    if (!newMachineName.trim()) {
      toast({
        title: "Validation Error",
        description: "Machine name cannot be empty",
        variant: "destructive",
      });
      return;
    }

    if (!newMachineKind) {
      toast({
        title: "Validation Error",
        description: "Machine kind is required",
        variant: "destructive",
      });
      return;
    }

    setSubmitting(true);

    try {
      // Cookie-based JWT is automatically sent with fetch requests
      const response = await fetch("/api/admin/machines", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({ 
          name: newMachineName.trim(),
          machine_kind: newMachineKind,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to create machine model");
      }

      toast({
        title: "Success",
        description: "Machine model created successfully",
      });

      setIsAddModalOpen(false);
      setNewMachineName("");
      setNewMachineKind("Print Engine");
      fetchMachines();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to create machine model";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  const handleEditMachine = (machine: MachineModel) => {
    setEditingMachine(machine);
    setEditingMachineName(machine.name);
    setEditingMachineKind(machine.machine_kind || "Print Engine");
    setIsEditModalOpen(true);
  };

  const handleUpdateMachine = async () => {
    if (!editingMachine) return;

    const trimmedName = editingMachineName.trim();
    if (!trimmedName) {
      toast({
        title: "Validation Error",
        description: "Machine name cannot be empty",
        variant: "destructive",
      });
      return;
    }

    if (!editingMachineKind) {
      toast({
        title: "Validation Error",
        description: "Machine kind is required",
        variant: "destructive",
      });
      return;
    }

    setSubmitting(true);

    try {
      const response = await fetch(`/api/admin/machines/${editingMachine.id}`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({ 
          name: trimmedName,
          machine_kind: editingMachineKind,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to update machine model");
      }

      toast({
        title: "Success",
        description: "Machine model updated successfully",
      });

      setIsEditModalOpen(false);
      setEditingMachine(null);
      fetchMachines();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to update machine model";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  const handleDeleteClick = (machine: MachineModel) => {
    setMachineToDelete(machine);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!machineToDelete) return;

    setDeleting(true);

    try {
      const response = await fetch(`/api/admin/machines/${machineToDelete.id}`, {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorCode = response.headers.get("X-Error-Code");
        
        // Handle 409 Conflict (machine model in use)
        if (response.status === 409 || errorCode === "MACHINE_MODEL_IN_USE") {
          throw new Error(
            errorData.detail || 
            "This machine model is in use and can't be deleted. Remove it from any machines/documents first."
          );
        }
        
        throw new Error(errorData.detail || "Failed to delete machine model");
      }

      toast({
        title: "Success",
        description: "Machine model deleted successfully. Associated documents have been set to NO MODEL.",
      });

      setIsDeleteModalOpen(false);
      setMachineToDelete(null);
      fetchMachines();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to delete machine model";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setDeleting(false);
    }
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return "—";
    try {
      const date = new Date(dateString);
      if (Number.isNaN(date.getTime())) return "—";
      return date.toLocaleString();
    } catch {
      return "—";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Machine Models</h1>
          <p className="text-sm text-muted-foreground mt-1">
            Manage machine models and view associated documents
          </p>
        </div>
        <Button onClick={() => setIsAddModalOpen(true)}>Add Machine Model</Button>
      </div>

      {/* Summary Stats */}
      {!loading && totalDocuments > 0 && (
        <div className="rounded-lg border border-border bg-muted/30 p-4">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-muted-foreground">Total Documents</div>
              <div className="text-lg font-semibold">{totalDocuments}</div>
            </div>
            <div>
              <div className="text-muted-foreground">Matched</div>
              <div className="text-lg font-semibold text-green-600">{matchedDocuments}</div>
            </div>
            {unmatchedDocuments > 0 && (
              <div>
                <div className="text-muted-foreground">Unmatched</div>
                <div className="text-lg font-semibold text-orange-600">{unmatchedDocuments}</div>
                {unmatchedMachineModels.length > 0 && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Models: {unmatchedMachineModels.join(", ")}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error Banner */}
      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Table Card */}
      <div className="rounded-lg border border-border bg-background">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">Loading machine models...</span>
          </div>
        ) : !Array.isArray(machines) || machines.length === 0 ? (
          <div className="py-12 text-center">
            <p className="text-sm text-muted-foreground">
              No machine models found.
              <br />
              Create one using the &quot;Add Machine Model&quot; button above.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border">
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Machine Name
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Machine Kind
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Documents
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Created
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {Array.isArray(machines) && machines.map((machine) => (
                  <tr key={machine.id} className="border-b border-border last:border-b-0">
                    <td className="px-4 py-3 text-sm font-medium">{machine.name}</td>
                    <td className="px-4 py-3 text-sm">
                      <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium bg-primary/10 text-primary">
                        {machine.machine_kind || "Print Engine"}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-muted-foreground">
                      {machine.document_count}
                    </td>
                    <td className="px-4 py-3 text-sm text-muted-foreground">
                      {formatDate(machine.created_at)}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <div className="flex items-center gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleEditMachine(machine)}
                          className="text-muted-foreground hover:text-foreground"
                          title="Edit machine model"
                        >
                          <Edit2 className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteClick(machine)}
                          className="text-muted-foreground hover:text-destructive"
                          title="Delete machine model"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Add Machine Modal */}
      {isAddModalOpen && (
        <Modal title="Create New Machine Model" onClose={() => setIsAddModalOpen(false)}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Machine Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={newMachineName}
                onChange={(e) => setNewMachineName(e.target.value)}
                placeholder="Enter machine model name"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !submitting) {
                    handleAddMachine();
                  }
                }}
                autoFocus
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Machine Kind <span className="text-red-500">*</span>
              </label>
              <select
                value={newMachineKind}
                onChange={(e) => setNewMachineKind(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                disabled={submitting}
              >
                {MACHINE_KINDS.map((kind) => (
                  <option key={kind} value={kind}>
                    {kind}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="mt-6 flex justify-end gap-3">
            <Button variant="outline" onClick={() => setIsAddModalOpen(false)} disabled={submitting}>
              Cancel
            </Button>
            <Button onClick={handleAddMachine} disabled={submitting || !newMachineName.trim() || !newMachineKind}>
              {submitting ? "Adding..." : "Add"}
            </Button>
          </div>
        </Modal>
      )}

      {/* Edit Machine Modal */}
      {isEditModalOpen && editingMachine && (
        <Modal title="Edit Machine Model" onClose={() => {
          setIsEditModalOpen(false);
          setEditingMachine(null);
        }}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Machine Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={editingMachineName}
                onChange={(e) => setEditingMachineName(e.target.value)}
                placeholder="Enter machine model name"
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                disabled={submitting}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !submitting && editingMachineName.trim() && editingMachineKind) {
                    handleUpdateMachine();
                  }
                }}
                autoFocus
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">
                Machine Kind <span className="text-red-500">*</span>
              </label>
              <select
                value={editingMachineKind}
                onChange={(e) => setEditingMachineKind(e.target.value)}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                disabled={submitting}
              >
                {MACHINE_KINDS.map((kind) => (
                  <option key={kind} value={kind}>
                    {kind}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="mt-6 flex justify-end gap-3">
            <Button 
              variant="outline" 
              onClick={() => {
                setIsEditModalOpen(false);
                setEditingMachine(null);
              }} 
              disabled={submitting}
            >
              Cancel
            </Button>
            <Button onClick={handleUpdateMachine} disabled={submitting || !editingMachineName.trim() || !editingMachineKind}>
              {submitting ? "Saving..." : "Save"}
            </Button>
          </div>
        </Modal>
      )}

      {/* Delete Confirmation Modal */}
      {isDeleteModalOpen && machineToDelete && (
        <Modal title="Delete Machine Model" onClose={() => {
          setIsDeleteModalOpen(false);
          setMachineToDelete(null);
        }}>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Are you sure you want to delete the machine model <strong>&quot;{machineToDelete.name}&quot;</strong>?
            </p>
            <p className="text-sm text-muted-foreground">
              This will remove the machine model and set any associated documents to NO MODEL. 
              You can create a new machine model and reassign documents afterward.
            </p>
          </div>
          <div className="mt-6 flex justify-end gap-3">
            <Button 
              variant="outline" 
              onClick={() => {
                setIsDeleteModalOpen(false);
                setMachineToDelete(null);
              }} 
              disabled={deleting}
            >
              Cancel
            </Button>
            <Button 
              variant="destructive"
              onClick={handleDeleteConfirm} 
              disabled={deleting}
            >
              {deleting ? "Deleting..." : "Delete"}
            </Button>
          </div>
        </Modal>
      )}
    </div>
  );
}


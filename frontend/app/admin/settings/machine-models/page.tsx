"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Trash2, Loader2 } from "lucide-react";

interface MachineModel {
  id: number;
  name: string;
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
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [machines, setMachines] = useState<MachineModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalDocuments, setTotalDocuments] = useState<number>(0);
  const [matchedDocuments, setMatchedDocuments] = useState<number>(0);
  const [unmatchedDocuments, setUnmatchedDocuments] = useState<number>(0);
  const [unmatchedMachineModels, setUnmatchedMachineModels] = useState<string[]>([]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [newMachineName, setNewMachineName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    const token = localStorage.getItem("auth_token");
    setAuthToken(token);
  }, []);

  const fetchMachines = useCallback(async () => {
    if (!authToken) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/admin/machines", {
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
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
  }, [authToken, toast]);

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

    if (!authToken) {
      toast({
        title: "Error",
        description: "Authentication required",
        variant: "destructive",
      });
      return;
    }

    setSubmitting(true);

    try {
      const response = await fetch("/api/admin/machines", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ name: newMachineName.trim() }),
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
                    <td className="px-4 py-3 text-sm text-muted-foreground">
                      {machine.document_count}
                    </td>
                    <td className="px-4 py-3 text-sm text-muted-foreground">
                      {formatDate(machine.created_at)}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <Button
                        variant="ghost"
                        size="sm"
                        disabled
                        className="text-muted-foreground"
                        title="Deletion will be available in a future update."
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
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
              <label className="block text-sm font-medium mb-2">Machine Name</label>
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
          </div>
          <div className="mt-6 flex justify-end gap-3">
            <Button variant="outline" onClick={() => setIsAddModalOpen(false)} disabled={submitting}>
              Cancel
            </Button>
            <Button onClick={handleAddMachine} disabled={submitting || !newMachineName.trim()}>
              {submitting ? "Adding..." : "Add"}
            </Button>
          </div>
        </Modal>
      )}
    </div>
  );
}


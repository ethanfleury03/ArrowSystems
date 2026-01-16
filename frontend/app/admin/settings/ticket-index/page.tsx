"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { RefreshCw, Loader2, CheckCircle2, AlertCircle } from "lucide-react";

interface TicketIndexStatus {
  manifest: {
    built_at: string;
    max_updated_at_indexed: string | null;
    eligible_count_indexed: number;
    index_prefix: string;
    job_execution_id: string | null;
  } | null;
  needs_reindex: boolean;
  reason: string | null;
  index_exists: boolean;
  db_stats: {
    max_updated_at: string | null;
    eligible_count: number;
  };
  active_execution: {
    name: string;
    state: string;
  } | null;
}

interface ReindexResponse {
  status: "triggered" | "up_to_date";
  execution_name: string | null;
  message: string;
}

export default function TicketIndexPage() {
  const [status, setStatus] = useState<TicketIndexStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [reindexing, setReindexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/admin/ticket-index/status", {
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to fetch ticket index status");
      }

      const data: TicketIndexStatus = await response.json();
      setStatus(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const triggerReindex = useCallback(async () => {
    setReindexing(true);
    setError(null);

    try {
      const response = await fetch("/api/admin/ticket-index/reindex", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || "Failed to trigger reindex");
      }

      const data: ReindexResponse = await response.json();

      if (data.status === "up_to_date") {
        toast({
          title: "Up to date",
          description: data.message,
        });
      } else {
        toast({
          title: "Reindex started",
          description: data.message,
        });
      }

      // Refresh status after a short delay
      setTimeout(() => {
        fetchStatus();
      }, 2000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Unknown error";
      setError(errorMessage);
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setReindexing(false);
    }
  }, [toast, fetchStatus]);

  useEffect(() => {
    fetchStatus();
    // Poll status every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "Never";
    try {
      const date = new Date(dateStr);
      return date.toLocaleString();
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="container mx-auto py-8 px-4 max-w-4xl">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold">Ticket Index Management</h1>
          <p className="text-muted-foreground mt-2">
            Manage ticket cache index rebuilds and monitor index status
          </p>
        </div>

        {error && (
          <div className="rounded-lg border border-destructive bg-destructive/10 p-4">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-destructive" />
              <p className="text-sm text-destructive">{error}</p>
            </div>
          </div>
        )}

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Index Status</h2>
            <Button
              variant="outline"
              size="sm"
              onClick={fetchStatus}
              disabled={loading}
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
            </Button>
          </div>

          {loading && !status ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : status ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Index Exists</p>
                  <p className="text-lg font-medium">
                    {status.index_exists ? (
                      <span className="text-green-600 flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5" />
                        Yes
                      </span>
                    ) : (
                      <span className="text-muted-foreground">No</span>
                    )}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Needs Reindex</p>
                  <p className="text-lg font-medium">
                    {status.needs_reindex ? (
                      <span className="text-orange-600">Yes</span>
                    ) : (
                      <span className="text-green-600 flex items-center gap-2">
                        <CheckCircle2 className="h-5 w-5" />
                        No
                      </span>
                    )}
                  </p>
                </div>
              </div>

              {status.manifest && (
                <div className="space-y-2 pt-4 border-t">
                  <h3 className="font-medium">Manifest Information</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Last Built</p>
                      <p className="font-medium">
                        {formatDate(status.manifest.built_at)}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Eligible Tickets Indexed</p>
                      <p className="font-medium">
                        {status.manifest.eligible_count_indexed}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Max Updated At (Indexed)</p>
                      <p className="font-medium">
                        {formatDate(status.manifest.max_updated_at_indexed)}
                      </p>
                    </div>
                    {status.manifest.job_execution_id && (
                      <div>
                        <p className="text-muted-foreground">Last Execution ID</p>
                        <p className="font-medium font-mono text-xs">
                          {status.manifest.job_execution_id}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="space-y-2 pt-4 border-t">
                <h3 className="font-medium">Database Statistics</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Eligible Tickets (DB)</p>
                    <p className="font-medium">{status.db_stats.eligible_count}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Max Updated At (DB)</p>
                    <p className="font-medium">
                      {formatDate(status.db_stats.max_updated_at)}
                    </p>
                  </div>
                </div>
              </div>

              {status.reason && (
                <div className="rounded-lg border border-orange-200 bg-orange-50 p-3">
                  <p className="text-sm text-orange-800">
                    <strong>Reason:</strong> {status.reason}
                  </p>
                </div>
              )}
            </div>
          ) : null}
        </div>

        <div className="rounded-lg border bg-card p-6">
          <h2 className="text-xl font-semibold mb-4">Actions</h2>
          <div className="space-y-4">
            <div>
              <p className="text-sm text-muted-foreground mb-2">
                Rebuild the ticket cache index from eligible tickets in the database.
                This will export artifacts, build a new index, and upload it to GCS.
              </p>
              <Button
                onClick={triggerReindex}
                disabled={reindexing || loading}
                className="w-full sm:w-auto"
              >
                {reindexing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Reindexing...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Re Index
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

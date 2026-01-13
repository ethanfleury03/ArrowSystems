"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Ticket, RefreshCw, Search, Edit, Eye, X } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface TicketItem {
  ticket_id: string;
  subject: string;
  status: string;
  created_at: string | null;
  updated_at: string | null;
  cache_eligible: boolean;
  review_status: string | null;
  manual_status: string | null;
  outcome: string | null;
  confidence: number;
  has_confirmation: boolean;
  machine_models: string[];
}

interface TicketsResponse {
  items: TicketItem[];
  page: number;
  page_size: number;
  total: number;
  cache_eligible_total: number;
}

type SortField = "ticket_id" | "judged_at" | "created_at" | "updated_at" | "cache_eligible" | "confidence";
type SortDirection = "asc" | "desc";

const formatDate = (dateStr: string | null | undefined): string => {
  if (!dateStr) return "—";
  try {
    return new Date(dateStr).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return dateStr;
  }
};

const getStatusBadgeClass = (status: string | null): string => {
  if (!status) return "border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400";
  const statusLower = status.toLowerCase();
  if (statusLower === "solved" || statusLower === "closed") {
    return "border-green-500/30 bg-green-500/10 text-green-700 dark:text-green-400";
  }
  if (statusLower === "open" || statusLower === "pending") {
    return "border-blue-500/30 bg-blue-500/10 text-blue-700 dark:text-blue-400";
  }
  return "border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400";
};

const getReviewStatusBadgeClass = (status: string | null): string => {
  if (!status) return "border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400";
  const statusLower = status.toLowerCase();
  if (statusLower === "approved") {
    return "border-green-500/30 bg-green-500/10 text-green-700 dark:text-green-400";
  }
  if (statusLower === "rejected") {
    return "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-400";
  }
  if (statusLower === "needs_review") {
    return "border-orange-500/30 bg-orange-500/10 text-orange-700 dark:text-orange-400";
  }
  return "border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400";
};

export default function AdminTicketsPage() {
  const [tickets, setTickets] = useState<TicketItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(50);
  const [total, setTotal] = useState(0);
  const [cacheEligibleTotal, setCacheEligibleTotal] = useState(0);
  const [sortField, setSortField] = useState<SortField>("judged_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingTicket, setEditingTicket] = useState<TicketItem | null>(null);
  const [editForm, setEditForm] = useState<Partial<TicketItem>>({});
  const [saving, setSaving] = useState(false);
  const [machineModels, setMachineModels] = useState<string[]>([]);
  const [loadingMachineModels, setLoadingMachineModels] = useState(false);
  const [selectedMachineModel, setSelectedMachineModel] = useState<string>("");
  const [viewDialogOpen, setViewDialogOpen] = useState(false);
  const [viewingTicketId, setViewingTicketId] = useState<string | null>(null);
  const [ticketDetails, setTicketDetails] = useState<any>(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [scrapeStatus, setScrapeStatus] = useState<any>(null);
  const [scraping, setScraping] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const { toast } = useToast();

  // Fetch machine models when dialog opens
  useEffect(() => {
    if (editDialogOpen && machineModels.length === 0) {
      setLoadingMachineModels(true);
      fetch("/api/admin/machine_models", { credentials: "include" })
        .then((res) => res.json())
        .then((data) => {
          setMachineModels(data.allowed_machine_models || []);
        })
        .catch((err) => {
          console.error("Failed to fetch machine models:", err);
          toast({
            title: "Warning",
            description: "Failed to load machine models. You can still type them manually.",
            variant: "destructive",
          });
        })
        .finally(() => {
          setLoadingMachineModels(false);
        });
    }
  }, [editDialogOpen, machineModels.length, toast]);

  const fetchTickets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const queryParams = new URLSearchParams({
        page: String(page),
        page_size: String(pageSize),
        sort: `${sortField} ${sortDirection}`,
        ...(searchTerm && { q: searchTerm }),
      });

      const response = await fetch(`/api/admin/tickets?${queryParams.toString()}`, {
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to fetch tickets" }));
        throw new Error(errorData.detail || `Failed to fetch tickets (${response.status})`);
      }

      const data: TicketsResponse = await response.json();
      setTickets(data.items);
      setTotal(data.total);
      setCacheEligibleTotal(data.cache_eligible_total || 0);
    } catch (err) {
      console.error("Failed to fetch tickets:", err);
      setError(err instanceof Error ? err.message : "Unable to load tickets.");
    } finally {
      setLoading(false);
    }
  }, [page, pageSize, searchTerm, sortField, sortDirection]);

  useEffect(() => {
    fetchTickets();
  }, [fetchTickets]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const handleEditClick = (ticket: TicketItem) => {
    setEditingTicket(ticket);
    setEditForm({
      subject: ticket.subject,
      status: ticket.status,
      cache_eligible: ticket.cache_eligible,
      confidence: ticket.confidence,
      review_status: ticket.review_status || null,
      outcome: ticket.outcome || null,
      machine_models: ticket.machine_models,
    });
    setSelectedMachineModel("");
    setEditDialogOpen(true);
  };

  const handleCancelScrape = async () => {
    try {
      const response = await fetch("/api/admin/scrape/cancel", {
        method: "POST",
        credentials: "include",
      });

      if (response.status === 401) {
        toast({
          title: "Authentication expired",
          description: "Please refresh the page and try again.",
          variant: "destructive",
        });
        return;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to cancel scrape" }));
        throw new Error(errorData.detail || "Failed to cancel scrape");
      }

      const data = await response.json();
      toast({
        title: "Scrape cancelled",
        description: data.message || "Scrape run has been cancelled.",
      });

      // Refresh status
      fetchScrapeStatus();
    } catch (err) {
      console.error("Failed to cancel scrape:", err);
      toast({
        title: "Cancel failed",
        description: err instanceof Error ? err.message : "Failed to cancel scrape",
        variant: "destructive",
      });
    }
  };

  const handleScrapeClick = async () => {
    setScraping(true);
    try {
      const response = await fetch("/api/admin/scrape/run", {
        method: "POST",
        credentials: "include",
      });

      if (response.status === 401) {
        // Token expired
        setScraping(false);
        toast({
          title: "Authentication expired",
          description: "Please refresh the page and try again.",
          variant: "destructive",
        });
        return;
      }

      if (response.status === 409) {
        // Already running - get status and start polling
        const errorData = await response.json().catch(() => ({ detail: "Scrape already running" }));
        toast({
          title: "Scrape already running",
          description: errorData.detail || "A scrape is already in progress.",
        });
        // Start polling for status
        startPollingStatus();
        return;
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to start scrape" }));
        throw new Error(errorData.detail || "Failed to start scrape");
      }

      const data = await response.json();
      toast({
        title: "Scrape started",
        description: `Scraping job ${data.run_id} has been started.`,
      });

      // Start polling for status
      startPollingStatus();
    } catch (err) {
      console.error("Failed to start scrape:", err);
      toast({
        title: "Scrape failed",
        description: err instanceof Error ? err.message : "Failed to start scrape",
        variant: "destructive",
      });
      setScraping(false);
    }
  };

  const fetchScrapeStatus = async () => {
    try {
      const response = await fetch("/api/admin/scrape/status", {
        credentials: "include",
      });

      if (!response.ok) {
        // Handle 401 (token expired) by stopping polling
        if (response.status === 401) {
          stopPollingStatus();
          setScraping(false);
          toast({
            title: "Authentication expired",
            description: "Please refresh the page to continue.",
            variant: "destructive",
          });
          return;
        }
        // For other errors, just return silently
        return;
      }

      const data = await response.json();
      setScrapeStatus(data);

      // Stop polling if completed/failed/cancelled/not_running
      if (data.status === "completed" || data.status === "failed" || data.status === "cancelled" || data.status === "not_running") {
        stopPollingStatus();
        setScraping(false);
        // Note: Table refresh is now manual-only via the Refresh button
      }
    } catch (err) {
      console.error("Failed to fetch scrape status:", err);
      // Stop polling on network errors
      stopPollingStatus();
      setScraping(false);
    }
  };

  const startPollingStatus = () => {
    // Clear any existing interval
    stopPollingStatus();

    // Fetch immediately
    fetchScrapeStatus();

    // Then poll every 2 seconds
    const interval = setInterval(() => {
      fetchScrapeStatus();
    }, 2000);

    setPollingInterval(interval);
  };

  const stopPollingStatus = () => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
      setPollingInterval(null);
    }
  };

  // Fetch initial scrape status on mount
  useEffect(() => {
    fetchScrapeStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingInterval) {
        clearInterval(pollingInterval);
      }
    };
  }, [pollingInterval]);

  // Auto-hide cancelled/completed status after a delay
  useEffect(() => {
    if (scrapeStatus?.status === "cancelled") {
      const timeout = setTimeout(() => {
        setScrapeStatus(null);
      }, 2500); // 2.5 seconds for cancelled

      return () => clearTimeout(timeout);
    } else if (scrapeStatus?.status === "completed") {
      const timeout = setTimeout(() => {
        setScrapeStatus(null);
      }, 5000); // 5 seconds for completed

      return () => clearTimeout(timeout);
    }
  }, [scrapeStatus?.status]);

  const handleViewClick = async (ticketId: string) => {
    setViewingTicketId(ticketId);
    setViewDialogOpen(true);
    setLoadingDetails(true);
    setTicketDetails(null);
    
    try {
      const response = await fetch(`/api/admin/tickets/${ticketId}`, {
        credentials: "include",
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to fetch ticket details" }));
        throw new Error(errorData.detail || "Failed to fetch ticket details");
      }
      
      const data = await response.json();
      setTicketDetails(data);
    } catch (err) {
      console.error("Failed to fetch ticket details:", err);
      toast({
        title: "Error",
        description: err instanceof Error ? err.message : "Failed to load ticket details",
        variant: "destructive",
      });
      setViewDialogOpen(false);
    } finally {
      setLoadingDetails(false);
    }
  };

  const handleSave = async () => {
    if (!editingTicket) return;

    setSaving(true);
    try {
      const response = await fetch(`/api/admin/tickets/${editingTicket.ticket_id}`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify({
          subject: editForm.subject,
          status: editForm.status,
          cache_eligible: editForm.cache_eligible,
          confidence: editForm.confidence,
          review_status: editForm.review_status,
          outcome: editForm.outcome,
          machine_models: editForm.machine_models,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Failed to update ticket" }));
        throw new Error(errorData.detail || "Failed to update ticket");
      }

      toast({
        title: "Ticket updated",
        description: `Ticket ${editingTicket.ticket_id} has been updated successfully.`,
      });

      setEditDialogOpen(false);
      setEditingTicket(null);
      // Refresh the tickets list
      fetchTickets();
    } catch (err) {
      console.error("Failed to update ticket:", err);
      toast({
        title: "Update failed",
        description: err instanceof Error ? err.message : "Failed to update ticket",
        variant: "destructive",
      });
    } finally {
      setSaving(false);
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 md:mx-0 md:px-6 xl:mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Ticket Management</h1>
          <p className="text-sm text-muted-foreground">
            View and manage cache-eligible support tickets from Zendesk.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {scrapeStatus?.status === "running" || scrapeStatus?.status === "pending" ? (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleCancelScrape}
              disabled={scraping}
            >
              Cancel
            </Button>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={handleScrapeClick}
              disabled={scraping || (scrapeStatus?.status === "running" || scrapeStatus?.status === "pending")}
            >
              Scrape
            </Button>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={fetchTickets}
            disabled={loading}
          >
            {loading ? (
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

      {/* Scrape Status Display */}
      {scrapeStatus && scrapeStatus.status !== "not_running" && (
        <div className="rounded-lg border bg-muted/50 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-semibold">Scrape Status:</span>
                  <span
                    className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
                      scrapeStatus.status === "completed"
                        ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        : scrapeStatus.status === "failed"
                        ? "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                        : scrapeStatus.status === "cancelled"
                        ? "bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200"
                        : scrapeStatus.status === "running" || scrapeStatus.status === "pending"
                        ? "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200"
                        : "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200"
                    }`}
                  >
                    {scrapeStatus.status}
                  </span>
                  {scrapeStatus.stage && (
                    <span className="text-xs text-muted-foreground">
                      ({scrapeStatus.stage})
                    </span>
                  )}
                </div>
                {scrapeStatus.summary && (
                  <div className="mt-2 text-xs text-muted-foreground">
                    {scrapeStatus.summary.tickets_indexed && (
                      <span>Indexed: {scrapeStatus.summary.tickets_indexed}</span>
                    )}
                    {scrapeStatus.summary.tickets_new !== undefined && (
                      <span className="ml-3">New: {scrapeStatus.summary.tickets_new}</span>
                    )}
                    {scrapeStatus.summary.tickets_detail_built !== undefined && (
                      <span className="ml-3">Built: {scrapeStatus.summary.tickets_detail_built}</span>
                    )}
                    {scrapeStatus.summary.tickets_judged !== undefined && (
                      <span className="ml-3">Judged: {scrapeStatus.summary.tickets_judged}</span>
                    )}
                  </div>
                )}
                {scrapeStatus.error && (
                  <div className="mt-2 text-xs text-red-600 dark:text-red-400">
                    Error: {scrapeStatus.error}
                  </div>
                )}
              </div>
            </div>
            {scrapeStatus.run_id && (
              <span className="text-xs text-muted-foreground font-mono">
                {scrapeStatus.run_id.substring(0, 8)}...
              </span>
            )}
          </div>
        </div>
      )}

      <section className="rounded-xl border bg-background shadow-sm">
        <div className="flex flex-col gap-4 border-b border-border p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold">Tickets</h2>
            <p className="text-xs text-muted-foreground">
              {total} total ticket{total !== 1 ? "s" : ""}
              {cacheEligibleTotal > 0 && (
                <span className="ml-2 text-green-600 dark:text-green-400">
                  • {cacheEligibleTotal} cache-eligible
                </span>
              )}
            </p>
          </div>
          <div className="flex w-full flex-col gap-2 md:w-auto">
            <span className="text-sm font-medium text-muted-foreground">Search</span>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <input
                type="text"
                placeholder="Search by ticket ID, subject, outcome..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  setPage(1); // Reset to first page on search
                }}
                className="w-full rounded-md border border-border bg-muted/70 px-3 py-2 pl-9 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary md:w-72"
              />
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border bg-muted/30">
                <th
                  className="cursor-pointer whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted/50"
                  onClick={() => handleSort("ticket_id")}
                >
                  Ticket ID
                  {sortField === "ticket_id" && (sortDirection === "asc" ? " ↑" : " ↓")}
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Subject
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Status
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Cache Eligible
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Review Status
                </th>
                <th
                  className="cursor-pointer whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted/50"
                  onClick={() => handleSort("confidence")}
                >
                  Confidence
                  {sortField === "confidence" && (sortDirection === "asc" ? " ↑" : " ↓")}
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Machine Models
                </th>
                <th
                  className="cursor-pointer whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:bg-muted/50"
                  onClick={() => handleSort("updated_at")}
                >
                  Updated
                  {sortField === "updated_at" && (sortDirection === "asc" ? " ↑" : " ↓")}
                </th>
                <th className="whitespace-nowrap px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-muted-foreground">
                    Loading tickets...
                  </td>
                </tr>
              ) : error ? (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-destructive">
                    {error}
                  </td>
                </tr>
              ) : tickets.length === 0 ? (
                <tr>
                  <td colSpan={9} className="px-4 py-6 text-center text-muted-foreground">
                    {searchTerm ? "No tickets match your search." : "No tickets found."}
                  </td>
                </tr>
              ) : (
                tickets.map((ticket) => (
                  <tr key={ticket.ticket_id} className="group transition-colors hover:bg-muted/40">
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-medium">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => handleEditClick(ticket)}
                          className="p-1 hover:bg-muted rounded transition-colors"
                          title="Edit ticket"
                        >
                          <Ticket className="h-4 w-4 text-muted-foreground hover:text-primary cursor-pointer" />
                        </button>
                        <span className="font-mono">{ticket.ticket_id}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className="max-w-xs truncate block" title={ticket.subject}>
                        {ticket.subject || "—"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span
                        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${getStatusBadgeClass(
                          ticket.status
                        )}`}
                      >
                        {ticket.status || "—"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span
                        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${
                          ticket.cache_eligible
                            ? "border-green-500/30 bg-green-500/10 text-green-700 dark:text-green-400"
                            : "border-gray-500/30 bg-gray-500/10 text-gray-700 dark:text-gray-400"
                        }`}
                      >
                        {ticket.cache_eligible ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span
                        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${getReviewStatusBadgeClass(
                          ticket.review_status || ticket.manual_status
                        )}`}
                      >
                        {ticket.review_status || ticket.manual_status || "—"}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                      {(ticket.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="px-4 py-3 text-sm text-muted-foreground">
                      {ticket.machine_models.length > 0 ? (
                        <span className="max-w-xs truncate block" title={ticket.machine_models.join(", ")}>
                          {ticket.machine_models.join(", ")}
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-muted-foreground">
                      {formatDate(ticket.updated_at)}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleViewClick(ticket.ticket_id)}
                        className="h-8"
                        title="View ticket details"
                      >
                        <Eye className="h-4 w-4 mr-1.5" />
                        View
                      </Button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-border px-4 py-3">
            <div className="text-sm text-muted-foreground">
              Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, total)} of {total} tickets
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1 || loading}
              >
                Previous
              </Button>
              <span className="text-sm text-muted-foreground">
                Page {page} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages || loading}
              >
                Next
              </Button>
            </div>
          </div>
        )}
      </section>

      {/* Edit Dialog */}
      <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
        <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Edit Ticket {editingTicket?.ticket_id}</DialogTitle>
            <DialogDescription>
              Update ticket fields. Leave fields empty to keep current values.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="subject">Subject</Label>
              <Input
                id="subject"
                value={editForm.subject || ""}
                onChange={(e) => setEditForm({ ...editForm, subject: e.target.value })}
                placeholder="Ticket subject"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="status">Status</Label>
              <Select
                value={editForm.status || "__none__"}
                onValueChange={(value) => setEditForm({ ...editForm, status: value === "__none__" ? undefined : value })}
              >
                <SelectTrigger id="status" className="w-full">
                  <SelectValue placeholder="Select status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">—</SelectItem>
                  <SelectItem value="new">New</SelectItem>
                  <SelectItem value="open">Open</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                  <SelectItem value="hold">Hold</SelectItem>
                  <SelectItem value="solved">Solved</SelectItem>
                  <SelectItem value="closed">Closed</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="cache_eligible" className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="cache_eligible"
                  checked={editForm.cache_eligible || false}
                  onChange={(e) => setEditForm({ ...editForm, cache_eligible: e.target.checked })}
                  className="h-4 w-4"
                />
                Cache Eligible
              </Label>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="confidence">Confidence (0.0 - 1.0)</Label>
              <Input
                id="confidence"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={editForm.confidence ?? ""}
                onChange={(e) =>
                  setEditForm({
                    ...editForm,
                    confidence: e.target.value ? parseFloat(e.target.value) : undefined,
                  })
                }
                placeholder="0.0 - 1.0"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="review_status">Review Status</Label>
              <Select
                value={editForm.review_status || "__none__"}
                onValueChange={(value) =>
                  setEditForm({
                    ...editForm,
                    review_status: value === "__none__" ? null : value,
                  })
                }
              >
                <SelectTrigger id="review_status" className="w-full">
                  <SelectValue placeholder="Select review status" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">—</SelectItem>
                  <SelectItem value="approved">Approved</SelectItem>
                  <SelectItem value="rejected">Rejected</SelectItem>
                  <SelectItem value="needs_review">Needs Review</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="outcome">Outcome</Label>
              <Input
                id="outcome"
                value={editForm.outcome || ""}
                onChange={(e) => setEditForm({ ...editForm, outcome: e.target.value })}
                placeholder="Ticket outcome"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="machine_models">Machine Models</Label>
              <div className="space-y-2">
                <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto p-2 border rounded-md bg-muted/30">
                  {editForm.machine_models && editForm.machine_models.length > 0 ? (
                    editForm.machine_models.map((model) => (
                      <span
                        key={model}
                        className="inline-flex items-center gap-1 rounded-full border border-primary/30 bg-primary/10 px-2 py-1 text-xs"
                      >
                        {model}
                        <button
                          type="button"
                          onClick={() =>
                            setEditForm({
                              ...editForm,
                              machine_models: editForm.machine_models?.filter((m) => m !== model) || [],
                            })
                          }
                          className="ml-1 hover:text-destructive"
                        >
                          ×
                        </button>
                      </span>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground">No machine models selected</span>
                  )}
                </div>
                <Select
                  value={selectedMachineModel || undefined}
                  onValueChange={(value) => {
                    if (value && value !== "__empty__" && value !== "__all_selected__" && !editForm.machine_models?.includes(value)) {
                      setEditForm({
                        ...editForm,
                        machine_models: [...(editForm.machine_models || []), value],
                      });
                      setSelectedMachineModel(""); // Reset select
                    }
                  }}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder={loadingMachineModels ? "Loading..." : "Add machine model"} />
                  </SelectTrigger>
                  <SelectContent>
                    {machineModels
                      .filter((model) => !editForm.machine_models?.includes(model))
                      .map((model) => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    {machineModels.length === 0 && !loadingMachineModels && (
                      <SelectItem value="__empty__" disabled>
                        No machine models available
                      </SelectItem>
                    )}
                    {machineModels.length > 0 &&
                      editForm.machine_models &&
                      editForm.machine_models.length === machineModels.length && (
                        <SelectItem value="__all_selected__" disabled>
                          All models selected
                        </SelectItem>
                      )}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setEditDialogOpen(false)} disabled={saving}>
              Cancel
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              {saving ? "Saving..." : "Apply"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* View Dialog */}
      <Dialog open={viewDialogOpen} onOpenChange={setViewDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div>
                <DialogTitle>Ticket {viewingTicketId}</DialogTitle>
                <DialogDescription>
                  {ticketDetails?.subject || "Loading ticket details..."}
                </DialogDescription>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setViewDialogOpen(false)}
                className="h-6 w-6"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </DialogHeader>

          {loadingDetails ? (
            <div className="py-8 text-center text-muted-foreground">
              Loading ticket details...
            </div>
          ) : ticketDetails ? (
            <div className="space-y-6 py-4">
              {/* Ticket Metadata */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-muted-foreground">Status:</span>{" "}
                  <span>{ticketDetails.status || "—"}</span>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">Cache Eligible:</span>{" "}
                  <span>{ticketDetails.cache_eligible ? "Yes" : "No"}</span>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">Confidence:</span>{" "}
                  <span>{(ticketDetails.confidence * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="font-medium text-muted-foreground">Created:</span>{" "}
                  <span>{formatDate(ticketDetails.created_at)}</span>
                </div>
                {ticketDetails.machine_models && ticketDetails.machine_models.length > 0 && (
                  <div className="col-span-2">
                    <span className="font-medium text-muted-foreground">Machine Models:</span>{" "}
                    <span>{ticketDetails.machine_models.join(", ")}</span>
                  </div>
                )}
              </div>

              {/* Outcome/Problem/Resolution */}
              {(ticketDetails.outcome || ticketDetails.problem || ticketDetails.resolution_steps) && (
                <div className="space-y-3 border-t pt-4">
                  {ticketDetails.problem && (
                    <div>
                      <h3 className="font-semibold text-sm mb-1">Problem</h3>
                      <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {typeof ticketDetails.problem === 'string'
                          ? ticketDetails.problem
                          : (ticketDetails.problem?.summary || ticketDetails.problem?.text || JSON.stringify(ticketDetails.problem))}
                      </p>
                    </div>
                  )}
                  {ticketDetails.outcome && (
                    <div>
                      <h3 className="font-semibold text-sm mb-1">Outcome</h3>
                      <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                        {typeof ticketDetails.outcome === 'string'
                          ? ticketDetails.outcome
                          : (ticketDetails.outcome?.summary || ticketDetails.outcome?.text || JSON.stringify(ticketDetails.outcome))}
                      </p>
                    </div>
                  )}
                  {ticketDetails.resolution_steps && (
                    <div>
                      <h3 className="font-semibold text-sm mb-1">Resolution Steps</h3>
                      {Array.isArray(ticketDetails.resolution_steps) ? (
                        <ol className="list-decimal list-inside space-y-1 text-sm text-muted-foreground">
                          {ticketDetails.resolution_steps.map((step: any, idx: number) => {
                            // Handle both string and object formats
                            const stepText = typeof step === 'string' 
                              ? step 
                              : (step?.summary || step?.text || JSON.stringify(step));
                            return (
                              <li key={idx} className="whitespace-pre-wrap">{stepText}</li>
                            );
                          })}
                        </ol>
                      ) : (
                        <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                          {typeof ticketDetails.resolution_steps === 'string'
                            ? ticketDetails.resolution_steps
                            : (ticketDetails.resolution_steps?.summary || JSON.stringify(ticketDetails.resolution_steps))}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Conversation */}
              {ticketDetails.conversation && (
                <div className="space-y-4 border-t pt-4">
                  <h3 className="font-semibold text-base">Conversation</h3>
                  
                  {/* Initial Description */}
                  {ticketDetails.conversation.request?.description && (
                    <div className="bg-muted/50 rounded-lg p-4">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs font-semibold text-muted-foreground uppercase">Initial Request</span>
                        {ticketDetails.conversation.request.created_at && (
                          <span className="text-xs text-muted-foreground">
                            {formatDate(ticketDetails.conversation.request.created_at)}
                          </span>
                        )}
                      </div>
                      <p className="text-sm whitespace-pre-wrap">{ticketDetails.conversation.request.description}</p>
                    </div>
                  )}

                  {/* Messages */}
                  {ticketDetails.conversation.messages && Array.isArray(ticketDetails.conversation.messages) && (
                    <div className="space-y-3">
                      {ticketDetails.conversation.messages
                        .sort((a: any, b: any) => {
                          const dateA = a.created_at || "";
                          const dateB = b.created_at || "";
                          return dateA.localeCompare(dateB);
                        })
                        .map((message: any, idx: number) => (
                          <div
                            key={idx}
                            className={`rounded-lg p-4 ${
                              message.role === "user" || message.role === "requester"
                                ? "bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900"
                                : "bg-muted/50"
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-2">
                              <span className="text-xs font-semibold text-muted-foreground uppercase">
                                {message.role === "user" || message.role === "requester" ? "Customer" : "Agent"}
                              </span>
                              {message.created_at && (
                                <span className="text-xs text-muted-foreground">
                                  {formatDate(message.created_at)}
                                </span>
                              )}
                            </div>
                            <p className="text-sm whitespace-pre-wrap">{message.text || message.body || "—"}</p>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              )}

              {!ticketDetails.conversation && (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  No conversation data available for this ticket.
                </div>
              )}
            </div>
          ) : (
            <div className="py-8 text-center text-destructive text-sm">
              Failed to load ticket details.
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

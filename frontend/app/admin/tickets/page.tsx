"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Ticket, RefreshCw, Search } from "lucide-react";

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
  const [sortField, setSortField] = useState<SortField>("judged_at");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");

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

      <section className="rounded-xl border bg-background shadow-sm">
        <div className="flex flex-col gap-4 border-b border-border p-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold">Tickets</h2>
            <p className="text-xs text-muted-foreground">
              {total} total ticket{total !== 1 ? "s" : ""}
              {tickets.filter((t) => t.cache_eligible).length > 0 && (
                <span className="ml-2 text-green-600 dark:text-green-400">
                  • {tickets.filter((t) => t.cache_eligible).length} cache-eligible
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
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-muted-foreground">
                    Loading tickets...
                  </td>
                </tr>
              ) : error ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-destructive">
                    {error}
                  </td>
                </tr>
              ) : tickets.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-6 text-center text-muted-foreground">
                    {searchTerm ? "No tickets match your search." : "No tickets found."}
                  </td>
                </tr>
              ) : (
                tickets.map((ticket) => (
                  <tr key={ticket.ticket_id} className="group transition-colors hover:bg-muted/40">
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-medium">
                      <div className="flex items-center gap-2">
                        <Ticket className="h-4 w-4 text-muted-foreground" />
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
    </div>
  );
}

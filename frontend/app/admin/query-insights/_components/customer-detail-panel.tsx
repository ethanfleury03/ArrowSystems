"use client";

import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { FilterChip } from "./filter-chip";
import { Skeleton } from "@/components/ui/skeleton";
import type { QueryInsightsCustomer, CustomerQuerySummary } from "@/types/queryInsights";

interface CustomerDetailPanelProps {
  customer: QueryInsightsCustomer | null;
  queries: CustomerQuerySummary[] | null;
  isLoading: boolean;
  roleFilter: "all" | "customer" | "technician";
  onRoleFilterChange: (value: "all" | "customer" | "technician") => void;
}

/**
 * Parse a backend date string, treating it as UTC if no timezone is present.
 */
function parseBackendDate(dateString: string): Date {
  if (!dateString) return new Date();
  const hasTZ = /[zZ]|[+\-]\d{2}:?\d{2}$/.test(dateString);
  const normalized = hasTZ ? dateString : dateString + "Z";
  return new Date(normalized);
}

function formatDateTime(dateString: string): string {
  const date = parseBackendDate(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function CustomerDetailPanel({
  customer,
  queries,
  isLoading,
  roleFilter,
  onRoleFilterChange,
}: CustomerDetailPanelProps) {
  if (!customer) {
    return (
      <div className="flex items-center justify-center h-full min-h-[320px] text-muted-foreground">
        <p className="text-sm">Select a customer to view their queries.</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="space-y-2">
          <Skeleton className="h-6 w-32" />
          <Skeleton className="h-4 w-24" />
        </div>
        <div className="flex gap-2">
          <Skeleton className="h-8 w-16" />
          <Skeleton className="h-8 w-20" />
          <Skeleton className="h-8 w-24" />
        </div>
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      </div>
    );
  }

  // Filter queries by role
  const filteredQueries =
    queries?.filter((q) => {
      if (roleFilter === "all") return true;
      if (roleFilter === "customer") return q.user_role === "CUSTOMER";
      if (roleFilter === "technician") return q.user_role === "TECHNICIAN";
      return true;
    }) ?? [];

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-semibold">{customer.name}</h3>
        <p className="text-sm text-muted-foreground">
          {queries?.length ?? 0} {queries?.length === 1 ? "query" : "queries"}
        </p>
      </div>

      <div className="flex gap-2 mb-3">
        <FilterChip
          label="All"
          value="all"
          current={roleFilter}
          onChange={onRoleFilterChange}
        />
        <FilterChip
          label="Customer"
          value="customer"
          current={roleFilter}
          onChange={onRoleFilterChange}
        />
        <FilterChip
          label="Technician"
          value="technician"
          current={roleFilter}
          onChange={onRoleFilterChange}
        />
      </div>

      <div className="space-y-3 max-h-[420px] overflow-y-auto pr-1">
        {filteredQueries.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No queries match this filter.
          </p>
        ) : (
          filteredQueries.map((q) => (
            <Link
              key={q.id}
              href={`/admin/query-insights/${customer.id}/${q.conversation_id}`}
            >
              <Card className="cursor-pointer hover:bg-muted/60 transition">
                <CardContent className="p-3 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800">
                      {q.user_role === "TECHNICIAN" ? "Technician" : "Customer"}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {formatDateTime(q.created_at)}
                    </span>
                  </div>
                  <p className="text-sm line-clamp-2">{q.query_text}</p>
                </CardContent>
              </Card>
            </Link>
          ))
        )}
      </div>
    </div>
  );
}


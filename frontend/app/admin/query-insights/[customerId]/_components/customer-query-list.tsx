"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState, useEffect } from "react";
import type { CustomerQueriesResponse } from "@/types/queryInsights";
import { Input } from "@/components/ui/input";

interface Props {
  data: CustomerQueriesResponse;
  initialSearch: string;
}

/**
 * Parse a backend date string, treating it as UTC if no timezone is present.
 * Backend may send naive UTC datetimes (e.g., "2025-12-08T15:10:00") which
 * should be interpreted as UTC, not local time.
 */
function parseBackendDate(dateString: string): Date {
  if (!dateString) return new Date();
  
  // If the string already has timezone info (Z or offset), use as-is
  const hasTZ = /[zZ]|[+\-]\d{2}:?\d{2}$/.test(dateString);
  const normalized = hasTZ ? dateString : dateString + "Z";
  
  return new Date(normalized);
}

function formatDate(dateString: string): string {
  const date = parseBackendDate(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function CustomerQueryList({ data, initialSearch }: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [search, setSearch] = useState(initialSearch);

  // Temporary deep log to verify data shape from API.
  useEffect(() => {
    console.log("ADMIN QUERY INSIGHTS – customer query list data", data);
  }, [data]);

  // Debug logging
  useEffect(() => {
    console.log("QueryInsights: customer page props", {
      customerId: data.customer_id,
      customerName: data.customer_name,
      totalQueries: data.total_queries,
      queriesLength: data.queries.length,
      lastQueryAt: data.last_query_at,
      queries: data.queries.slice(0, 3), // First 3 queries for inspection
    });
  }, [data]);

  // Update URL query param when search changes (debounced).
  useEffect(() => {
    const timeout = setTimeout(() => {
      const params = new URLSearchParams(searchParams?.toString() ?? "");
      if (search) params.set("search", search);
      else params.delete("search");

      router.replace(
        `/admin/query-insights/${data.customer_id}?${params.toString()}`,
        { scroll: false }
      );
    }, 300);

    return () => clearTimeout(timeout);
  }, [search, searchParams, router, data.customer_id]);

  return (
    <div className="flex flex-col h-full">
      <header className="flex items-center justify-between gap-4 p-4 border-b">
        <div>
          <button
            type="button"
            onClick={() => router.push("/admin/query-insights")}
            className="text-sm text-muted-foreground hover:underline"
          >
            ← All customers
          </button>
          <h1 className="text-xl font-semibold mt-1">{data.customer_name}</h1>
          <p className="text-xs text-muted-foreground">
            {data.total_queries} queries
            {data.last_query_at
              ? ` · Last active ${formatDate(data.last_query_at)}`
              : ""}
          </p>
        </div>
        <div className="w-full max-w-xs">
          <Input
            placeholder="Search queries..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </header>

      <main className="flex-1 overflow-auto">
        {data.queries.length === 0 ? (
          <div className="h-full flex items-center justify-center text-muted-foreground">
            No queries found.
          </div>
        ) : (
          <ul className="divide-y">
            {data.queries.map((q) => (
              <li key={q.id}>
                <button
                  type="button"
                  className="w-full text-left px-4 py-3 hover:bg-muted/70 transition-colors"
                  onClick={() =>
                    router.push(
                      `/admin/query-insights/${data.customer_id}/${q.conversation_id}`
                    )
                  }
                >
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-sm font-medium truncate">
                      {q.query_text}
                    </p>
                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                      {formatDate(q.created_at)}
                    </span>
                  </div>
                  {typeof q.message_count === "number" && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {q.message_count} messages
                    </p>
                  )}
                </button>
              </li>
            ))}
          </ul>
        )}
      </main>
    </div>
  );
}


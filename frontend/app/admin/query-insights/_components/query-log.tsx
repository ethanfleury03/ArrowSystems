"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import type { RecentQueryLogItem } from "@/types/queryInsights";

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

export function QueryLog() {
  const [items, setItems] = useState<RecentQueryLogItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await fetch("/api/admin/query-insights/recent-queries?limit=50", {
          cache: "no-store",
        });
        if (!res.ok) throw new Error("Failed to load query log");
        const data: RecentQueryLogItem[] = await res.json();
        setItems(data);
      } catch (error) {
        console.error("Failed to load query log:", error);
        setItems([]);
      } finally {
        setIsLoading(false);
      }
    }
    load();
  }, []);

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-4 space-y-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex items-start gap-3">
              <Skeleton className="h-2 w-2 rounded-full mt-1" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-full" />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-4 space-y-3">
        {items.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No recent queries found.
          </p>
        ) : (
          items.map((item) => (
            <Link
              key={item.id}
              href={`/admin/query-insights/${String(item.customer_id)}/${item.conversation_id}`}
            >
              <div className="flex items-start gap-3 rounded-md px-2 py-2 hover:bg-muted/70 cursor-pointer transition">
                <div className="mt-1 h-2 w-2 rounded-full bg-emerald-500" />
                <div className="flex-1 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-medium px-1.5 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800">
                        {item.customer_name}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {item.user_role === "TECHNICIAN" ? "Technician" : "Customer"} ·{" "}
                        {item.user_email}
                      </span>
                      {item.machine_name && (
                        <span className="text-xs text-muted-foreground">
                          · {item.machine_name}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {formatDateTime(item.created_at)}
                    </span>
                  </div>
                  <p className="text-sm text-foreground line-clamp-2">
                    {item.query_text}
                  </p>
                </div>
              </div>
            </Link>
          ))
        )}
      </CardContent>
    </Card>
  );
}


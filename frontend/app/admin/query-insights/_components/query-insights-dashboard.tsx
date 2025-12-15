"use client";

import { useState, useEffect, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { UserBubbleMap } from "./user-bubble-map";
import { CustomerDetailPanel } from "./customer-detail-panel";
import type { QueryInsightsCustomer, CustomerQuerySummary, CustomerQueriesResponse, UserInsightPoint } from "@/types/queryInsights";
import { fetchUserInsights, type UserInsightResponse } from "@/lib/api/queryInsights";

interface QueryInsightsDashboardProps {
  initialCustomers: QueryInsightsCustomer[];
}

function parseBackendDate(dateString: string | null): Date {
  if (!dateString) return new Date();
  const hasTZ = /[zZ]|[+\-]\d{2}:?\d{2}$/.test(dateString);
  const normalized = hasTZ ? dateString : dateString + "Z";
  return new Date(normalized);
}

function mapUserInsightToPoint(insight: UserInsightResponse): UserInsightPoint {
  const role = insight.role?.toLowerCase() || "unknown";
  const normalizedRole = role === "customer" ? "customer" : role === "technician" ? "technician" : "unknown";
  
  const lastActiveMs = insight.last_query_at
    ? parseBackendDate(insight.last_query_at).getTime()
    : Date.now() - 86400000 * 365; // Default to 1 year ago if no last query

  return {
    userId: insight.user_id,
    email: insight.email || "unknown@example.com",
    role: normalizedRole as "customer" | "technician" | "unknown",
    totalQueries: insight.total_queries || 0,
    queries7d: insight.queries_7d || 0,
    lastActiveMs,
    name: insight.name || insight.email || "Unknown",
  };
}

export function QueryInsightsDashboard({ initialCustomers }: QueryInsightsDashboardProps) {
  const [userInsights, setUserInsights] = useState<UserInsightPoint[]>([]);
  const [isLoadingInsights, setIsLoadingInsights] = useState(true);
  const [selectedUser, setSelectedUser] = useState<UserInsightPoint | null>(null);
  const [selectedCustomer, setSelectedCustomer] = useState<QueryInsightsCustomer | null>(null);
  const [queries, setQueries] = useState<CustomerQuerySummary[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [roleFilter, setRoleFilter] = useState<"all" | "customer" | "technician">("all");

  // Fetch user insights on mount
  useEffect(() => {
    let mounted = true;
    async function loadUserInsights() {
      setIsLoadingInsights(true);
      try {
        const insights = await fetchUserInsights();
        const points = insights.map(mapUserInsightToPoint);
        if (!mounted) return;
        
        setUserInsights(points);
        
        // Auto-select most active user if no selection and data exists
        if (points.length > 0) {
          const mostActive = points.reduce((max, point) => 
            point.totalQueries > max.totalQueries ? point : max
          );
          setSelectedUser(mostActive);
        }
      } catch (error) {
        console.error("Failed to fetch user insights:", error);
        if (mounted) {
          setUserInsights([]);
        }
      } finally {
        if (mounted) {
          setIsLoadingInsights(false);
        }
      }
    }
    loadUserInsights();
    return () => {
      mounted = false;
    };
  }, []);

  // Map selected user to customer for sidebar and fetch queries
  useEffect(() => {
    if (selectedUser) {
      // Find matching customer from initialCustomers
      const matchingCustomer = initialCustomers.find(
        (c) => c.id === selectedUser.userId
      );
      
      if (matchingCustomer) {
        setSelectedCustomer(matchingCustomer);
      } else {
        // If user is not in customers list (e.g., technician), create a synthetic customer object
        setSelectedCustomer({
          id: selectedUser.userId,
          name: selectedUser.name,
          total_queries: selectedUser.totalQueries,
          last_query_at: new Date(selectedUser.lastActiveMs).toISOString(),
        });
      }
      
      // Fetch queries for this user
      setIsLoading(true);
      fetch(`/api/admin/query-insights/customers/${selectedUser.userId}/queries`, {
        cache: "no-store",
      })
        .then((res) => {
          if (!res.ok) {
            // If user is not a customer, we might not have queries endpoint for them
            setQueries([]);
            return;
          }
          return res.json();
        })
        .then((data: CustomerQueriesResponse | undefined) => {
          if (data) {
            setQueries(data.queries ?? []);
          }
        })
        .catch((error) => {
          console.error("Failed to fetch user queries:", error);
          setQueries([]);
        })
        .finally(() => {
          setIsLoading(false);
        });
    } else {
      setSelectedCustomer(null);
      setQueries(null);
    }
  }, [selectedUser, initialCustomers]);

  function handleSelectUser(point: UserInsightPoint) {
    setSelectedUser(point);
    // Query fetching is handled by useEffect when selectedUser changes
  }

  return (
    <Card className="w-full">
      <CardContent className="p-6 flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-h-[320px]">
          {isLoadingInsights ? (
            <div className="flex items-center justify-center min-h-[320px] text-muted-foreground">
              <p className="text-sm">Loading user insights...</p>
            </div>
          ) : (
            <UserBubbleMap
              points={userInsights}
              selectedUserId={selectedUser?.userId ?? null}
              onSelectUser={handleSelectUser}
              roleFilter={roleFilter}
            />
          )}
        </div>
        <div className="w-full lg:w-[380px] xl:w-[420px] lg:border-l lg:pl-6 lg:mt-0 mt-4 pt-4 lg:pt-0">
          <CustomerDetailPanel
            customer={selectedCustomer}
            queries={queries}
            isLoading={isLoading}
            roleFilter={roleFilter}
            onRoleFilterChange={setRoleFilter}
          />
        </div>
      </CardContent>
    </Card>
  );
}


"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { resolveApiBaseUrl } from "@/config/api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface AnalyticsData {
  queriesOverTime: { date: string; query_count: number }[];
  queriesPerUser: { user_id: number; email: string; query_count: number }[];
  queriesByMachine: { machine_name: string; query_count: number }[];
  tokenUsage: {
    buckets: { date: string; token_input: number; token_output: number; token_total: number; cost_usd: number }[];
    totals: { token_input: number; token_output: number; token_total: number; cost_usd: number };
  };
  tokenUsagePerUser: { user_id: number; email: string; token_total: number; cost_usd: number }[];
  documentUsage: { document_id: string; display_name: string; usage_count: number }[];
  topKeywords: { keyword: string; count: number }[];
}

export default function AdminAnalyticsPage() {
  const router = useRouter();
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [isLoadingAnalytics, setIsLoadingAnalytics] = useState(false);
  const [analyticsError, setAnalyticsError] = useState<string | null>(null);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsData | null>(null);

  // Filter state
  const [dateRange, setDateRange] = useState<"last_7_days" | "last_30_days" | "custom">("last_7_days");
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [selectedUserId, setSelectedUserId] = useState<string>("");
  const [selectedMachine, setSelectedMachine] = useState<string>("");

  const apiBaseUrl = useMemo(() => resolveApiBaseUrl(), []);

  // Auth guard
  useEffect(() => {
    try {
      const token = localStorage.getItem("auth_token");
      if (!token) {
        throw new Error("Missing token");
      }

      const payloadBase64 = token.split(".")[1];
      if (!payloadBase64) {
        throw new Error("Invalid token payload");
      }
      const payloadJson = atob(payloadBase64.replace(/-/g, "+").replace(/_/g, "/"));
      const payload = JSON.parse(payloadJson);

      if (!payload?.role || payload.role !== "ADMIN") {
        throw new Error("Not an admin");
      }

      setAuthToken(token);
      setIsAdmin(true);
    } catch (error) {
      setIsAdmin(false);
      router.replace("/");
    }
  }, [router]);

  // Calculate date range
  const getDateRange = useCallback(() => {
    const end = new Date();
    end.setHours(23, 59, 59, 999);
    let start: Date;

    if (dateRange === "last_7_days") {
      start = new Date();
      start.setDate(start.getDate() - 7);
      start.setHours(0, 0, 0, 0);
    } else if (dateRange === "last_30_days") {
      start = new Date();
      start.setDate(start.getDate() - 30);
      start.setHours(0, 0, 0, 0);
    } else {
      // Custom range
      if (startDate && endDate) {
        start = new Date(startDate);
        start.setHours(0, 0, 0, 0);
        end.setTime(new Date(endDate).getTime());
        end.setHours(23, 59, 59, 999);
      } else {
        // Default to last 7 days if custom but no dates selected
        start = new Date();
        start.setDate(start.getDate() - 7);
        start.setHours(0, 0, 0, 0);
      }
    }

    return {
      start_date: start.toISOString().split("T")[0],
      end_date: end.toISOString().split("T")[0],
    };
  }, [dateRange, startDate, endDate]);

  // Fetch analytics data
  const fetchAnalytics = useCallback(async () => {
    if (!authToken) return;

    setIsLoadingAnalytics(true);
    setAnalyticsError(null);

    try {
      const { start_date, end_date } = getDateRange();
      const params = new URLSearchParams({
        start_date,
        end_date,
      });

      if (selectedUserId) {
        params.append("user_id", selectedUserId);
      }
      if (selectedMachine) {
        params.append("machine_name", selectedMachine);
      }

      const baseUrl = `${apiBaseUrl}/admin/analytics`;
      const [queriesOverTime, queriesPerUser, queriesByMachine, tokenUsage, tokenUsagePerUser, documentUsage, topKeywords] =
        await Promise.all([
          fetch(`${baseUrl}/queries_over_time?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
          fetch(`${baseUrl}/queries_per_user?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
          fetch(`${baseUrl}/queries_by_machine?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
          fetch(`${baseUrl}/token_usage?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
          fetch(`${baseUrl}/token_usage_per_user?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
          fetch(`${baseUrl}/document_usage?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
          fetch(`${baseUrl}/top_keywords?${params}`, {
            headers: { Authorization: `Bearer ${authToken}` },
          }).then((r) => r.json()),
        ]);

      setAnalyticsData({
        queriesOverTime: queriesOverTime.buckets || [],
        queriesPerUser: queriesPerUser.items || [],
        queriesByMachine: queriesByMachine.items || [],
        tokenUsage: tokenUsage || { buckets: [], totals: { token_input: 0, token_output: 0, token_total: 0, cost_usd: 0 } },
        tokenUsagePerUser: tokenUsagePerUser.items || [],
        documentUsage: documentUsage.items || [],
        topKeywords: topKeywords.items || [],
      });
    } catch (err) {
      console.error("Failed to fetch analytics:", err);
      setAnalyticsError(err instanceof Error ? err.message : "Failed to load analytics data");
    } finally {
      setIsLoadingAnalytics(false);
    }
  }, [authToken, apiBaseUrl, getDateRange, selectedUserId, selectedMachine]);

  // Fetch on mount and when filters change
  useEffect(() => {
    if (isAdmin && authToken) {
      fetchAnalytics();
    }
  }, [isAdmin, authToken, fetchAnalytics]);

  // Calculate KPIs
  const kpis = useMemo(() => {
    if (!analyticsData) return null;

    const totalQueries = analyticsData.queriesOverTime.reduce((sum, b) => sum + b.query_count, 0);
    const totalTokens = analyticsData.tokenUsage.totals.token_total || 0;
    const totalCost = analyticsData.tokenUsage.totals.cost_usd || 0;
    const uniqueUsers = new Set(analyticsData.queriesPerUser.map((u) => u.user_id)).size;
    const avgQueriesPerDay =
      analyticsData.queriesOverTime.length > 0
        ? totalQueries / analyticsData.queriesOverTime.length
        : 0;
    const estimatedMonthlyCost = totalCost * (30 / (dateRange === "last_7_days" ? 7 : 30));

    return {
      totalQueries,
      totalTokens,
      totalCost,
      uniqueUsers,
      avgQueriesPerDay,
      estimatedMonthlyCost,
    };
  }, [analyticsData, dateRange]);

  if (isAdmin === null) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground">Validating access...</p>
      </div>
    );
  }

  if (!isAdmin) {
    return null;
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 md:px-6">
      <div>
        <h1 className="text-2xl font-semibold">Analytics Dashboard</h1>
        <p className="text-sm text-muted-foreground">View query metrics, token usage, and cost analytics</p>
      </div>

      {/* Filters */}
      <div className="rounded-xl border bg-background p-4 shadow-sm">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="grid gap-2">
            <label className="text-sm font-medium text-muted-foreground">Date Range</label>
            <select
              value={dateRange}
              onChange={(e) => setDateRange(e.target.value as typeof dateRange)}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
            >
              <option value="last_7_days">Last 7 Days</option>
              <option value="last_30_days">Last 30 Days</option>
              <option value="custom">Custom Range</option>
            </select>
          </div>

          {dateRange === "custom" && (
            <>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                />
              </div>
              <div className="grid gap-2">
                <label className="text-sm font-medium text-muted-foreground">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                />
              </div>
            </>
          )}

          <div className="grid gap-2">
            <label className="text-sm font-medium text-muted-foreground">Machine (Optional)</label>
            <input
              type="text"
              value={selectedMachine}
              onChange={(e) => setSelectedMachine(e.target.value)}
              placeholder="Filter by machine..."
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-primary focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>
      </div>

      {/* Error Message */}
      {analyticsError && (
        <div className="rounded-md border border-destructive bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {analyticsError}
        </div>
      )}

      {/* Loading State */}
      {isLoadingAnalytics && (
        <div className="flex items-center justify-center py-12">
          <p className="text-muted-foreground">Loading analytics...</p>
        </div>
      )}

      {/* KPI Cards */}
      {!isLoadingAnalytics && kpis && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <div className="text-sm font-medium text-muted-foreground">Total Queries</div>
            <div className="mt-2 text-3xl font-bold">{kpis.totalQueries.toLocaleString()}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              Avg: {kpis.avgQueriesPerDay.toFixed(1)} per day
            </div>
          </div>

          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <div className="text-sm font-medium text-muted-foreground">Total Tokens</div>
            <div className="mt-2 text-3xl font-bold">{(kpis.totalTokens / 1000).toFixed(1)}K</div>
            <div className="mt-1 text-xs text-muted-foreground">
              {kpis.totalTokens.toLocaleString()} tokens
            </div>
          </div>

          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <div className="text-sm font-medium text-muted-foreground">Total Cost</div>
            <div className="mt-2 text-3xl font-bold">${kpis.totalCost.toFixed(2)}</div>
            <div className="mt-1 text-xs text-muted-foreground">
              Est. monthly: ${kpis.estimatedMonthlyCost.toFixed(2)}
            </div>
          </div>

          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <div className="text-sm font-medium text-muted-foreground">Active Users</div>
            <div className="mt-2 text-3xl font-bold">{kpis.uniqueUsers}</div>
            <div className="mt-1 text-xs text-muted-foreground">Unique users in period</div>
          </div>
        </div>
      )}

      {/* Charts */}
      {!isLoadingAnalytics && analyticsData && (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Queries Over Time */}
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Queries Over Time</h2>
            {analyticsData.queriesOverTime.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analyticsData.queriesOverTime}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="query_count" stroke="#8884d8" name="Queries" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data in this period
              </div>
            )}
          </div>

          {/* Token Usage Over Time */}
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Token Usage Over Time</h2>
            {analyticsData.tokenUsage.buckets.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analyticsData.tokenUsage.buckets}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="token_input" stroke="#82ca9d" name="Input Tokens" />
                  <Line type="monotone" dataKey="token_output" stroke="#ffc658" name="Output Tokens" />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data in this period
              </div>
            )}
          </div>

          {/* Queries Per User */}
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Queries Per User</h2>
            {analyticsData.queriesPerUser.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.queriesPerUser.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="email" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="query_count" fill="#8884d8" name="Queries" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data in this period
              </div>
            )}
          </div>

          {/* Queries By Machine */}
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Queries By Machine</h2>
            {analyticsData.queriesByMachine.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.queriesByMachine}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="machine_name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="query_count" fill="#82ca9d" name="Queries" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data in this period
              </div>
            )}
          </div>

          {/* Token Usage Per User */}
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Token Usage Per User</h2>
            {analyticsData.tokenUsagePerUser.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.tokenUsagePerUser.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="email" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="token_total" fill="#ffc658" name="Total Tokens" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data in this period
              </div>
            )}
          </div>

          {/* Top Keywords */}
          <div className="rounded-xl border bg-background p-6 shadow-sm">
            <h2 className="mb-4 text-lg font-semibold">Top Keywords</h2>
            {analyticsData.topKeywords.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analyticsData.topKeywords.slice(0, 10)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="keyword" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="count" fill="#ff7c7c" name="Count" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-[300px] items-center justify-center text-muted-foreground">
                No data in this period
              </div>
            )}
          </div>
        </div>
      )}

      {/* Document Usage Table */}
      {!isLoadingAnalytics && analyticsData && analyticsData.documentUsage.length > 0 && (
        <div className="rounded-xl border bg-background p-6 shadow-sm">
          <h2 className="mb-4 text-lg font-semibold">Most Used Documents</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-border">
              <thead className="bg-muted/30">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Document
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Usage Count
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border bg-background">
                {analyticsData.documentUsage.slice(0, 20).map((doc, idx) => (
                  <tr key={idx}>
                    <td className="whitespace-nowrap px-4 py-3 text-sm">{doc.display_name}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm">{doc.usage_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}




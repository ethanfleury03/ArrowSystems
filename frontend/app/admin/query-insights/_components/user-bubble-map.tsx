"use client";

import { useMemo, useState } from "react";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { UserInsightPoint } from "@/types/queryInsights";

interface UserBubbleMapProps {
  points: UserInsightPoint[];
  selectedUserId: string | null;
  onSelectUser?: (point: UserInsightPoint) => void;
  roleFilter?: "all" | "customer" | "technician";
}

// Color mapping for roles
const ROLE_COLORS: Record<string, string> = {
  customer: "#3b82f6", // blue
  technician: "#10b981", // green
  unknown: "#6b7280", // gray
};

const ROLE_LABELS: Record<string, string> = {
  customer: "Customer",
  technician: "Technician",
  unknown: "Unknown",
};

function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatRecency(hours: number): string {
  if (hours < 1) return "<1h";
  if (hours < 24) return `${Math.round(hours)}h`;
  return `${Math.round(hours / 24)}d`;
}

// Custom tooltip component
const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length > 0) {
    const data = payload[0].payload as UserInsightPoint;
    return (
      <div className="bg-background border rounded-lg shadow-lg p-3 space-y-1">
        <p className="font-medium">{data.email}</p>
        <p className="text-xs text-muted-foreground">
          Role: {ROLE_LABELS[data.role] || data.role}
        </p>
        <p className="text-xs text-muted-foreground">
          Total queries: {data.totalQueries}
        </p>
        <p className="text-xs text-muted-foreground">
          Queries (7d): {data.queries7d}
        </p>
        <p className="text-xs text-muted-foreground">
          Last active: {formatDate(data.lastActiveMs)}
        </p>
      </div>
    );
  }
  return null;
};

// Custom legend component
const CustomLegend = ({ payload }: any) => {
  if (!payload || payload.length === 0) return null;
  
  const uniqueRoles = Array.from(
    new Set(payload.map((entry: any) => entry.payload?.role || "unknown"))
  ) as string[];

  return (
    <div className="flex justify-center gap-4">
      {uniqueRoles.map((role) => (
        <div key={role} className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: ROLE_COLORS[role] || ROLE_COLORS.unknown }}
          />
          <span className="text-xs text-muted-foreground">
            {ROLE_LABELS[role] || role}
          </span>
        </div>
      ))}
    </div>
  );
};

export function UserBubbleMap({
  points,
  selectedUserId,
  onSelectUser,
  roleFilter = "all",
}: UserBubbleMapProps) {
  // Filter points by role
  const filteredPoints = useMemo(() => {
    if (roleFilter === "all") return points;
    if (roleFilter === "customer") {
      return points.filter((p) => p.role === "customer");
    }
    if (roleFilter === "technician") {
      return points.filter((p) => p.role === "technician");
    }
    return points;
  }, [points, roleFilter]);

  // Calculate domains with padding (X is now hours since last active)
  const { xDomain, yDomain, zDomain } = useMemo(() => {
    if (filteredPoints.length === 0) {
      return {
        xDomain: [0, 168], // 0 to 7 days in hours
        yDomain: [0, 100],
        zDomain: [0, 100],
      };
    }

    const now = Date.now();
    // Calculate hours since last active for each point
    const xValues = filteredPoints
      .map((p) => Math.max(0, (now - p.lastActiveMs) / (1000 * 60 * 60)))
      .filter((v) => !isNaN(v) && isFinite(v));
    const yValues = filteredPoints.map((p) => p.totalQueries);
    const zValues = filteredPoints.map((p) => p.queries7d);

    const xMax = Math.max(...xValues, 1); // At least 1 hour
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const zMin = Math.min(...zValues);
    const zMax = Math.max(...zValues);

    // Add padding (10% on top, 0 on bottom since 0 is "now")
    const xPadding = xMax * 0.1 || 1;
    const yPadding = (yMax - yMin) * 0.1 || 10;
    const zPadding = (zMax - zMin) * 0.1 || 1;

    return {
      xDomain: [0, Math.max(1, xMax + xPadding)], // 0 = now (most recent)
      yDomain: [Math.max(0, yMin - yPadding), yMax + yPadding],
      zDomain: [Math.max(0, zMin - zPadding), zMax + zPadding],
    };
  }, [filteredPoints]);

  // Prepare data for Recharts (needs x, y, z format)
  // X is now hours since last active (0 = now, larger = older)
  const chartData = useMemo(() => {
    const now = Date.now();
    return filteredPoints.map((point) => {
      const hoursSinceActive = Math.max(0, (now - point.lastActiveMs) / (1000 * 60 * 60));
      return {
        ...point,
        x: hoursSinceActive,
        y: point.totalQueries,
        z: Math.max(1, point.queries7d), // Ensure minimum size
      };
    });
  }, [filteredPoints]);

  // Group by role for separate scatter series
  const dataByRole = useMemo(() => {
    const grouped: Record<string, typeof chartData> = {
      customer: [],
      technician: [],
      unknown: [],
    };

    chartData.forEach((point) => {
      const role = point.role || "unknown";
      if (grouped[role]) {
        grouped[role].push(point);
      } else {
        grouped.unknown.push(point);
      }
    });

    return grouped;
  }, [chartData]);

  if (filteredPoints.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <div className="text-center space-y-2">
          <p className="text-sm font-medium">No user activity yet.</p>
          <p className="text-xs">
            User query activity will appear here once they start using the system.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart
            margin={{ top: 10, right: 20, bottom: 40, left: 20 }}
            onClick={(data) => {
              if (data?.activePayload?.[0]?.payload) {
                const point = data.activePayload[0].payload as UserInsightPoint;
                onSelectUser?.(point);
              }
            }}
          >
            <XAxis
              type="number"
              dataKey="x"
              domain={xDomain}
              reversed={true}
              tickFormatter={(value) => formatRecency(value)}
              label={{ value: "Hours Since Last Active", position: "insideBottom", offset: -5 }}
            />
          <YAxis
            type="number"
            dataKey="y"
            domain={yDomain}
            label={{ value: "Total Queries", angle: -90, position: "insideLeft" }}
          />
          <ZAxis type="number" dataKey="z" range={[20, 200]} />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            content={<CustomLegend />}
            verticalAlign="top"
            align="center"
            wrapperStyle={{ paddingTop: "10px" }}
          />
          
          {/* Render separate scatter series for each role */}
          {Object.entries(dataByRole).map(([role, data]) => {
            if (data.length === 0) return null;
            return (
              <Scatter
                key={role}
                name={ROLE_LABELS[role] || role}
                data={data}
                fill={ROLE_COLORS[role] || ROLE_COLORS.unknown}
              >
                {data.map((entry, index) => {
                  const isSelected = entry.userId === selectedUserId;
                  return (
                    <Cell
                      key={`cell-${index}`}
                      fill={ROLE_COLORS[role] || ROLE_COLORS.unknown}
                      stroke={isSelected ? "#000" : "transparent"}
                      strokeWidth={isSelected ? 3 : 0}
                      style={{ cursor: "pointer" }}
                    />
                  );
                })}
              </Scatter>
            );
          })}
        </ScatterChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}


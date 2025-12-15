"use client";

import { useMemo } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { QueryInsightsCustomer } from "@/types/queryInsights";

interface CustomerGalaxyProps {
  customers: QueryInsightsCustomer[];
  selectedCustomerId: string | null;
  onSelectCustomer?: (customer: QueryInsightsCustomer) => void;
}

interface PositionedCustomer extends QueryInsightsCustomer {
  x: number;
  y: number;
  radius: number;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function mapQueriesToRadius(totalQueries: number): number {
  // Map query count to radius: 0 queries = 32px, scale up to 80px for high query counts
  if (totalQueries === 0) return 32;
  // Logarithmic scaling: log(queries + 1) * scale factor
  const logValue = Math.log10(totalQueries + 1);
  const maxLog = Math.log10(1000); // Assume 1000 queries is "high"
  const normalized = logValue / maxLog;
  return 32 + normalized * 48; // 32 to 80 range
}

function formatDate(dateString: string | null): string {
  if (!dateString) return "Never";
  const date = new Date(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function CustomerGalaxy({ customers, selectedCustomerId, onSelectCustomer }: CustomerGalaxyProps) {
  const positioned = useMemo<PositionedCustomer[]>(() => {
    const centerX = 50;
    const centerY = 50;
    const baseRadius = 10;
    const ringSpacing = 8;

    const total = customers.length;
    if (total === 0) return [];

    return customers.map((cust, index) => {
      const angle = (2 * Math.PI * index) / total;
      const ring = index % 4;
      const radius = baseRadius + ring * ringSpacing;
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);
      
      // Calculate circle radius based on total_queries
      const circleRadius = clamp(mapQueriesToRadius(cust.total_queries), 32, 80);

      return { ...cust, x, y, radius: circleRadius };
    });
  }, [customers]);

  if (positioned.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[320px] text-muted-foreground">
        <div className="text-center space-y-2">
          <p className="text-sm font-medium">No customers with queries yet.</p>
          <p className="text-xs">Customer query activity will appear here once they start using the system.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-[320px] rounded-xl bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950 shadow-inner overflow-hidden">
      <div className="absolute inset-0">
        {positioned.map((cust) => {
          const isSelected = cust.id === selectedCustomerId;
          return (
            <Tooltip key={cust.id}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className={`absolute flex items-center justify-center rounded-full border bg-background shadow-md text-xs font-medium px-2 text-center leading-tight transition-all cursor-pointer ${
                    isSelected
                      ? "ring-2 ring-primary ring-offset-2 border-primary scale-110"
                      : "hover:scale-110"
                  }`}
                  style={{
                    left: `${cust.x}%`,
                    top: `${cust.y}%`,
                    width: `${cust.radius}px`,
                    height: `${cust.radius}px`,
                    transform: "translate(-50%, -50%)",
                    padding: "8px",
                  }}
                  onClick={() => onSelectCustomer?.(cust)}
                >
                  <span className="truncate w-full">{cust.name}</span>
                </button>
              </TooltipTrigger>
              <TooltipContent>
                <div className="space-y-1">
                  <p className="font-medium">{cust.name}</p>
                  <p className="text-xs">Total queries: {cust.total_queries}</p>
                  <p className="text-xs">Last active: {formatDate(cust.last_query_at)}</p>
                </div>
              </TooltipContent>
            </Tooltip>
          );
        })}
      </div>
    </div>
  );
}



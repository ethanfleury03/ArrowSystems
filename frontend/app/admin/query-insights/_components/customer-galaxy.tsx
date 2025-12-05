"use client";

import { useMemo } from "react";
import { useRouter } from "next/navigation";
import type { QueryInsightsCustomer } from "@/types/queryInsights";

interface CustomerGalaxyProps {
  customers: QueryInsightsCustomer[];
}

interface PositionedCustomer extends QueryInsightsCustomer {
  x: number;
  y: number;
}

export function CustomerGalaxy({ customers }: CustomerGalaxyProps) {
  const router = useRouter();

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

      return { ...cust, x, y };
    });
  }, [customers]);

  if (positioned.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted-foreground">
        No customers with queries yet.
      </div>
    );
  }

  return (
    <div className="flex-1 relative overflow-hidden bg-gradient-to-b from-background to-muted">
      <div className="absolute inset-0">
        {positioned.map((cust) => {
          const initials = getInitials(cust.name);
          return (
            <button
              key={cust.id}
              type="button"
              className="absolute flex items-center justify-center rounded-full border bg-background shadow-md text-sm font-medium transition-transform hover:scale-110"
              style={{
                left: `${cust.x}%`,
                top: `${cust.y}%`,
                width: "64px",
                height: "64px",
                transform: "translate(-50%, -50%)",
              }}
              onClick={() => router.push(`/admin/query-insights/${cust.id}`)}
            >
              {initials}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) {
    return parts[0].slice(0, 2).toUpperCase();
  }
  return (parts[0][0] + parts[1][0]).toUpperCase();
}


"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface FilterChipProps {
  label: string;
  value: "all" | "customer" | "technician";
  current: "all" | "customer" | "technician";
  onChange: (value: "all" | "customer" | "technician") => void;
}

export function FilterChip({ label, value, current, onChange }: FilterChipProps) {
  const isActive = value === current;

  return (
    <Button
      type="button"
      variant={isActive ? "default" : "outline"}
      size="sm"
      className={cn(
        "h-8 rounded-full px-3 text-xs",
        isActive && "bg-primary text-primary-foreground"
      )}
      onClick={() => onChange(value)}
    >
      {label}
    </Button>
  );
}


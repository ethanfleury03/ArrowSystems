"use client";

import { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { CustomerGalaxy } from "./customer-galaxy";
import { CustomerDetailPanel } from "./customer-detail-panel";
import type { QueryInsightsCustomer, CustomerQuerySummary, CustomerQueriesResponse } from "@/types/queryInsights";

interface QueryInsightsDashboardProps {
  initialCustomers: QueryInsightsCustomer[];
}

export function QueryInsightsDashboard({ initialCustomers }: QueryInsightsDashboardProps) {
  const [selectedCustomer, setSelectedCustomer] = useState<QueryInsightsCustomer | null>(null);
  const [queries, setQueries] = useState<CustomerQuerySummary[] | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [roleFilter, setRoleFilter] = useState<"all" | "customer" | "technician">("all");

  async function handleSelectCustomer(customer: QueryInsightsCustomer) {
    setSelectedCustomer(customer);
    setIsLoading(true);
    try {
      const res = await fetch(
        `/api/admin/query-insights/customers/${customer.id}/queries`,
        { cache: "no-store" }
      );
      if (!res.ok) throw new Error("Failed to load queries");
      const data: CustomerQueriesResponse = await res.json();
      setQueries(data.queries ?? []);
    } catch (error) {
      console.error("Failed to fetch customer queries:", error);
      setQueries([]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <Card className="w-full">
      <CardContent className="p-6 flex flex-col lg:flex-row gap-6">
        <div className="flex-1 min-h-[320px]">
          <CustomerGalaxy
            customers={initialCustomers}
            selectedCustomerId={selectedCustomer?.id ?? null}
            onSelectCustomer={handleSelectCustomer}
          />
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


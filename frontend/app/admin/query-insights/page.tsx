import { fetchQueryInsightsCustomers } from "@/lib/api/queryInsights";
import { SectionHeader } from "./_components/section-header";
import { QueryInsightsDashboard } from "./_components/query-insights-dashboard";
import { QueryLog } from "./_components/query-log";
import type { QueryInsightsCustomer } from "@/types/queryInsights";

export const dynamic = 'force-dynamic';

export default async function QueryInsightsPage() {
  let customers: QueryInsightsCustomer[] = [];
  let error: string | null = null;

  try {
    customers = await fetchQueryInsightsCustomers();
  } catch (e) {
    console.error('Failed to fetch query insights customers:', e);
    error = e instanceof Error ? e.message : 'Failed to load customers';
  }

  return (
    <div className="space-y-6">
      {error ? (
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center space-y-2">
            <p className="text-sm font-medium text-destructive">Error loading customers</p>
            <p className="text-xs text-muted-foreground">{error}</p>
          </div>
        </div>
      ) : (
        <>
          <SectionHeader
            title="Query Insights"
            description="Explore customer and technician query activity."
          />
          <QueryInsightsDashboard initialCustomers={customers} />
          <SectionHeader
            title="Query Log"
            description="Recent queries across all customers."
          />
          <QueryLog />
        </>
      )}
    </div>
  );
}


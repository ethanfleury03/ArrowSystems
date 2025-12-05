import { fetchQueryInsightsCustomers } from "@/lib/api/queryInsights";
import { CustomerGalaxy } from "./_components/customer-galaxy";
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
    <div className="flex flex-col min-h-[calc(100vh-8rem)]">
      <header className="flex items-center justify-between p-4 border-b bg-background rounded-t-lg">
        <div>
          <h1 className="text-xl font-semibold">Query Insights</h1>
          <p className="text-sm text-muted-foreground">
            Explore customer query activity.
          </p>
        </div>
        <div className="text-sm text-muted-foreground">
          {customers.length} {customers.length === 1 ? 'customer' : 'customers'}
        </div>
      </header>
      <div className="flex-1 bg-background rounded-b-lg border border-t-0">
        {error ? (
          <div className="flex items-center justify-center min-h-[400px]">
            <div className="text-center space-y-2">
              <p className="text-sm font-medium text-destructive">Error loading customers</p>
              <p className="text-xs text-muted-foreground">{error}</p>
            </div>
          </div>
        ) : (
          <CustomerGalaxy customers={customers} />
        )}
      </div>
    </div>
  );
}


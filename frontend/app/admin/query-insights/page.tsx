import { fetchQueryInsightsCustomers } from "@/lib/api/queryInsights";
import { CustomerGalaxy } from "./_components/customer-galaxy";

export default async function QueryInsightsPage() {
  const customers = await fetchQueryInsightsCustomers();

  return (
    <div className="flex flex-col h-full">
      <header className="flex items-center justify-between p-4 border-b">
        <div>
          <h1 className="text-xl font-semibold">Query Insights</h1>
          <p className="text-sm text-muted-foreground">
            Explore customer query activity.
          </p>
        </div>
        <div className="text-sm text-muted-foreground">
          {customers.length} customers
        </div>
      </header>
      <CustomerGalaxy customers={customers} />
    </div>
  );
}


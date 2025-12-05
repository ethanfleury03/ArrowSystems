import {
  QueryInsightsCustomer,
  CustomerQueriesResponse,
  ConversationDetails,
} from "@/types/queryInsights";

export async function fetchQueryInsightsCustomers(): Promise<QueryInsightsCustomer[]> {
  const res = await fetch("/api/admin/query-insights/customers", {
    credentials: "include",
  });
  if (!res.ok) throw new Error("Failed to load customers");
  return res.json();
}

export async function fetchCustomerQueries(
  customerId: string,
  search?: string
): Promise<CustomerQueriesResponse> {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  const res = await fetch(
    `/api/admin/query-insights/customers/${customerId}/queries?${params.toString()}`,
    { credentials: "include" }
  );
  if (!res.ok) throw new Error("Failed to load customer queries");
  return res.json();
}

export async function fetchConversationDetails(
  conversationId: string
): Promise<ConversationDetails> {
  const res = await fetch(
    `/api/admin/query-insights/conversations/${conversationId}`,
    { credentials: "include" }
  );
  if (!res.ok) throw new Error("Failed to load conversation");
  return res.json();
}


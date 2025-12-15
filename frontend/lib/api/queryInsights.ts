import { headers as getHeaders } from "next/headers";
import { cookies as getCookies } from "next/headers";
import {
  QueryInsightsCustomer,
  CustomerQueriesResponse,
  ConversationDetails,
} from "@/types/queryInsights";

// NOTE: This module is for **server-side** data fetching only.
// Do NOT import it from client components, because it uses next/headers.

async function getServerFetchOptions(): Promise<{ baseUrl: string; headers: Record<string, string> }> {
  // Server-side: get base URL and cookies
  try {
    const headersList = getHeaders();
    const cookieStore = getCookies();
    const host = headersList.get("host");
    const protocol =
      headersList.get("x-forwarded-proto") ||
      (process.env.NODE_ENV === "production" ? "https" : "http");

    const baseUrl = host
      ? `${protocol}://${host}`
      : process.env.VERCEL_URL
        ? `https://${process.env.VERCEL_URL}`
        : "http://localhost:3000";

    // Get all cookies and format as Cookie header
    const allCookies = cookieStore.getAll();
    const cookieHeader = allCookies.map((c) => `${c.name}=${c.value}`).join("; ");

    const fetchHeaders: Record<string, string> = {};
    if (cookieHeader) {
      fetchHeaders.Cookie = cookieHeader;
    }

    return {
      baseUrl,
      headers: fetchHeaders,
    };
  } catch {
    // Fallback if headers/cookies fail (e.g., during build validation)
    return {
      baseUrl: process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : "http://localhost:3000",
      headers: {},
    };
  }
}

export async function fetchQueryInsightsCustomers(): Promise<QueryInsightsCustomer[]> {
  const { baseUrl, headers: fetchHeaders } = await getServerFetchOptions();
  const url = baseUrl
    ? `${baseUrl}/api/admin/query-insights/customers`
    : "/api/admin/query-insights/customers";

  const res = await fetch(url, {
    credentials: "include",
    headers: fetchHeaders,
    cache: "no-store", // Ensure fresh data
  });

  if (!res.ok) {
    let errorMessage = "Failed to load customers";
    try {
      const errorData = await res.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch {
      errorMessage = `Failed to load customers: ${res.status} ${res.statusText}`;
    }
    throw new Error(errorMessage);
  }

  const data = await res.json();
  // Ensure we return an array even if backend returns unexpected format
  if (!Array.isArray(data)) {
    console.warn("Query Insights API returned non-array data:", data);
    return [];
  }
  return data;
}

export async function fetchCustomerQueries(
  customerId: string,
  search?: string,
): Promise<CustomerQueriesResponse> {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  const { baseUrl, headers: fetchHeaders } = await getServerFetchOptions();
  const url = baseUrl
    ? `${baseUrl}/api/admin/query-insights/customers/${customerId}/queries?${params.toString()}`
    : `/api/admin/query-insights/customers/${customerId}/queries?${params.toString()}`;

  const res = await fetch(url, {
    credentials: "include",
    headers: fetchHeaders,
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to load customer queries");
  return res.json();
}

export async function fetchConversationDetails(
  conversationId: string,
): Promise<ConversationDetails> {
  const { baseUrl, headers: fetchHeaders } = await getServerFetchOptions();
  const url = baseUrl
    ? `${baseUrl}/api/admin/query-insights/conversations/${conversationId}`
    : `/api/admin/query-insights/conversations/${conversationId}`;

  const res = await fetch(url, {
    credentials: "include",
    headers: fetchHeaders,
    cache: "no-store",
  });
  if (!res.ok) throw new Error("Failed to load conversation");
  return res.json();
}

export interface UserInsightResponse {
  user_id: string;
  email: string;
  name: string;
  role: string;
  total_queries: number;
  queries_7d: number;
  last_query_at: string | null;
}

export async function fetchUserInsights(): Promise<UserInsightResponse[]> {
  const { baseUrl, headers: fetchHeaders } = await getServerFetchOptions();
  const url = baseUrl 
    ? `${baseUrl}/api/admin/query-insights/users`
    : "/api/admin/query-insights/users";
  
  const res = await fetch(url, {
    credentials: "include",
    headers: fetchHeaders,
    cache: "no-store",
  });
  
  if (!res.ok) {
    let errorMessage = "Failed to load user insights";
    try {
      const errorData = await res.json();
      errorMessage = errorData.detail || errorData.message || errorMessage;
    } catch {
      errorMessage = `Failed to load user insights: ${res.status} ${res.statusText}`;
    }
    throw new Error(errorMessage);
  }
  
  const data = await res.json();
  if (!Array.isArray(data)) {
    console.warn('User insights API returned non-array data:', data);
    return [];
  }
  return data;
}


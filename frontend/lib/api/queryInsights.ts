import { headers as getHeaders } from "next/headers";
import { cookies as getCookies } from "next/headers";
import {
  QueryInsightsCustomer,
  CustomerQueriesResponse,
  ConversationDetails,
} from "@/types/queryInsights";

async function getServerFetchOptions(): Promise<{ baseUrl: string; headers: Record<string, string> }> {
  if (typeof window !== "undefined") {
    // Client-side: return empty options, browser handles cookies
    return { baseUrl: "", headers: {} };
  }
  
  // Server-side: get base URL and cookies
  try {
    const headersList = getHeaders();
    const cookieStore = getCookies();
    const host = headersList.get("host");
    const protocol = headersList.get("x-forwarded-proto") || 
                     (process.env.NODE_ENV === "production" ? "https" : "http");
    
    const baseUrl = host ? `${protocol}://${host}` : 
                    (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 
                     "http://localhost:3000");
    
    // Get all cookies and format as Cookie header
    const allCookies = cookieStore.getAll();
    const cookieHeader = allCookies.map(c => `${c.name}=${c.value}`).join("; ");
    
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
    // This should not happen with force-dynamic, but handle gracefully
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
  if (!res.ok) throw new Error("Failed to load customers");
  return res.json();
}

export async function fetchCustomerQueries(
  customerId: string,
  search?: string
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
  conversationId: string
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


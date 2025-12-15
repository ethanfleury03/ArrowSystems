export interface QueryInsightsCustomer {
  id: string;
  name: string;
  total_queries: number;
  last_query_at: string | null;
}

export interface CustomerQuerySummary {
  id: string;
  conversation_id: string;
  created_at: string;
  query_text: string;
  message_count?: number;

  // Who asked this query
  user_id: number;
  user_email: string;
  user_role: string;
}

export interface CustomerQueriesResponse {
  customer_id: string;
  customer_name: string;
  total_queries: number;
  last_query_at: string | null;
  queries: CustomerQuerySummary[];
}

export interface ConversationMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationDetails {
  conversation_id: string;
  customer_id: string;
  customer_name: string;
  created_at: string;
  messages: ConversationMessage[];
}


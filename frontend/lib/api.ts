import axios from 'axios';

// Use Next.js API route as proxy (works in both dev and Docker)
// The API route handles proxying to the backend
const getApiBaseUrl = () => {
  // Always use relative path to Next.js API route
  // Next.js API route will proxy to backend
  return '/api';
};

const apiClient = axios.create({
  baseURL: getApiBaseUrl(),
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SourceInfo {
  id: string;
  name: string;
  pages: string;
  content_type: string;
}

export interface DocumentSource {
  doc_id: string;
  pages_used: number[];
  snippet?: string;  // Short extract/snippet (~200 chars) for quick relevance check
}

export interface SummarizationInfo {
  was_summarized: boolean;
  content_type?: string;
  original_length: number;
  summarized_length: number;
}

export interface QueryResponse {
  query: string;
  answer: string;
  reasoning?: string;
  sources?: SourceInfo[];
  document_sources?: DocumentSource[];
  confidence?: number;
  intent_type?: string;
  intent_confidence?: number;
  response_time_ms?: number;
  session_id?: string;
  cache_hit?: boolean;
  matched_machine_name?: string;
  summarization_info?: SummarizationInfo;
  is_saved?: boolean;
}

export interface QueryParams {
  top_k?: number;
  alpha?: number;
  dynamic_windowing?: boolean;
}

export async function sendQuery(query: string, params?: QueryParams): Promise<QueryResponse> {
  try {
    const response = await apiClient.post<QueryResponse>('/query', {
      query,
      top_k: params?.top_k ?? 10,
      alpha: params?.alpha ?? 0.5,
      dynamic_windowing: params?.dynamic_windowing ?? true,
    });
    
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to get response';
      throw new Error(errorMessage);
    }
    throw error;
  }
}

export async function getHealth(): Promise<boolean> {
  try {
    const response = await apiClient.get('/health');
    return response.data.status === 'healthy';
  } catch {
    return false;
  }
}

export interface ChatHistoryItem {
  id: string;
  query: string;
  answer: string;
  timestamp: string;
  intent_type?: string;
  confidence?: number;
  sources?: string[];
  response_time_ms?: number;
}

export interface ChatHistoryResponse {
  status: string;
  count: number;
  history: ChatHistoryItem[];
}

export async function getChatHistory(user: string = 'api_user', limit: number = 50): Promise<ChatHistoryResponse> {
  try {
    const response = await apiClient.get<ChatHistoryResponse>(`/history?user=${user}&limit=${limit}`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to get chat history';
      throw new Error(errorMessage);
    }
    throw error;
  }
}

export interface SavedResponse {
  id: string;
  query: string;
  answer: string;
  sources: string[];
  helpful_count: number;
  unhelpful_count: number;
  last_used: string;
  first_validated: string;
  created_at: string;
}

export interface SavedResponsesResponse {
  status: string;
  count: number;
  saved: SavedResponse[];
}

export async function getSavedResponses(limit: number = 50, minHelpfulCount: number = 1, user: string = 'api_user'): Promise<SavedResponsesResponse> {
  try {
    const response = await apiClient.get<SavedResponsesResponse>(`/saved?user=${user}&limit=${limit}&min_helpful_count=${minHelpfulCount}`);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to get saved responses';
      throw new Error(errorMessage);
    }
    throw error;
  }
}

export interface FeedbackPayload {
  query: string;
  answer: string;
  is_helpful: boolean;
  session_id?: string;
  reasoning?: string;
  sources?: SourceInfo[];
  document_sources?: DocumentSource[];
  confidence?: number;
  intent_type?: string;
  intent_confidence?: number;
  matched_machine_name?: string;
  top_k?: number;
  alpha?: number;
  user?: string;
}

export interface FeedbackResponse {
  status: string;
  saved_to_file: boolean;
  saved_to_db: boolean;
  cache_updated: boolean;
  message?: string;
}

export async function submitFeedback(payload: FeedbackPayload): Promise<FeedbackResponse> {
  try {
    const response = await apiClient.post<FeedbackResponse>('/feedback', payload);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to submit feedback';
      throw new Error(errorMessage);
    }
    throw error;
  }
}

export interface SaveResponsePayload {
  query: string;
  answer: string;
  is_saved: boolean;
  session_id?: string;
  reasoning?: string;
  sources?: SourceInfo[];
  document_sources?: DocumentSource[];
  confidence?: number;
  intent_type?: string;
  intent_confidence?: number;
  matched_machine_name?: string;
  top_k?: number;
  alpha?: number;
  user?: string;
}

export interface SaveResponseResult {
  status: string;
  is_saved: boolean;
  saved_to_file: boolean;
  saved_to_db: boolean;
  cache_updated: boolean;
  message?: string;
}

export async function toggleSavedResponse(payload: SaveResponsePayload): Promise<SaveResponseResult> {
  try {
    const response = await apiClient.post<SaveResponseResult>('/saved', payload);
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to save response';
      throw new Error(errorMessage);
    }
    throw error;
  }
}


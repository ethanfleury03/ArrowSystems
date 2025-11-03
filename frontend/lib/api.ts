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

export interface QueryResponse {
  query: string;
  answer: string;
  reasoning?: string;
  sources?: Array<{
    id: string;
    name: string;
    pages: string;
    content_type: string;
  }>;
  confidence?: number;
  intent_type?: string;
  intent_confidence?: number;
  response_time_ms?: number;
  cache_hit?: boolean;
}

export async function sendQuery(query: string): Promise<string> {
  try {
    const response = await apiClient.post<QueryResponse>('/query', {
      query,
      top_k: 10,
      alpha: 0.5,
      dynamic_windowing: true,
    });
    
    return response.data.answer;
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


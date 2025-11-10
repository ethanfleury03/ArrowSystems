import type { DocumentSource, SourceInfo } from '@/lib/api';

export interface MessageSource {
  id: string;
  title: string;
  snippet: string;
  url?: string;
}

export interface AssistantMetadata {
  query: string;
  reasoning?: string;
  structuredSources: SourceInfo[];
  documentSources?: DocumentSource[];
  confidence?: number;
  intentType?: string;
  intentConfidence?: number;
  sessionId?: string;
  topK: number;
  alpha: number;
  matchedMachineName?: string;
  isSaved?: boolean;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: MessageSource[];
  metadata?: AssistantMetadata;
}

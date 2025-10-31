export interface Message {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp?: Date;
}


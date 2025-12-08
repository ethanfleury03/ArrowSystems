"use client";

import { useRouter } from "next/navigation";
import type { ConversationDetails } from "@/types/queryInsights";

interface Props {
  conversation: ConversationDetails;
}

/**
 * Parse a backend date string, treating it as UTC if no timezone is present.
 * Backend may send naive UTC datetimes (e.g., "2025-12-08T15:10:00") which
 * should be interpreted as UTC, not local time.
 */
function parseBackendDate(dateString: string): Date {
  if (!dateString) return new Date();
  
  // If the string already has timezone info (Z or offset), use as-is
  const hasTZ = /[zZ]|[+\-]\d{2}:?\d{2}$/.test(dateString);
  const normalized = hasTZ ? dateString : dateString + "Z";
  
  return new Date(normalized);
}

function formatDate(dateString: string): string {
  const date = parseBackendDate(dateString);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function ConversationView({ conversation }: Props) {
  const router = useRouter();

  return (
    <div className="flex flex-col h-full">
      <header className="flex items-center justify-between p-4 border-b gap-4">
        <div>
          <button
            type="button"
            onClick={() =>
              router.push(
                `/admin/query-insights/${conversation.customer_id}`
              )
            }
            className="text-sm text-muted-foreground hover:underline"
          >
            ← Back to {conversation.customer_name} queries
          </button>
          <h1 className="text-lg font-semibold mt-1">
            Conversation with {conversation.customer_name}
          </h1>
          <p className="text-xs text-muted-foreground">
            Started {formatDate(conversation.created_at)}
          </p>
        </div>
      </header>

      <main className="flex-1 overflow-auto p-4 space-y-4">
        {conversation.messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex ${
              msg.role === "user" ? "justify-start" : "justify-end"
            }`}
          >
            <div
              className={`max-w-xl rounded-lg px-3 py-2 text-sm shadow-sm ${
                msg.role === "user"
                  ? "bg-muted text-foreground"
                  : "bg-primary text-primary-foreground"
              }`}
            >
              <p className="whitespace-pre-wrap break-words">
                {msg.content}
              </p>
              <p className="mt-1 text-[10px] opacity-70 text-right">
                {formatDate(msg.created_at)}
              </p>
            </div>
          </div>
        ))}
      </main>
    </div>
  );
}


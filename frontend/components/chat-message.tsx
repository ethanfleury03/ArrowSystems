"use client"

import { User, Sparkles, ThumbsUp, ThumbsDown, Bookmark, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { useState } from "react"
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type Source = {
  id: string
  title: string
  snippet: string
  url?: string
}

type Message = {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  sources?: Source[]
}

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null)
  const [isSaved, setIsSaved] = useState(false)

  const handleFeedback = (type: "up" | "down") => {
    setFeedback(feedback === type ? null : type)
    // TODO: Send feedback to backend
    console.log(`Feedback: ${type} for message ${message.id}`)
  }

  const handleSave = () => {
    setIsSaved(!isSaved)
    // TODO: Save response to backend
    console.log(`${isSaved ? "Unsaved" : "Saved"} message ${message.id}`)
  }

  return (
    <div className={cn("flex items-start gap-4", isUser && "flex-row-reverse")}>
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
          isUser ? "bg-primary" : "bg-accent",
        )}
      >
        {isUser ? (
          <User className="h-4 w-4 text-primary-foreground" />
        ) : (
          <Sparkles className="h-4 w-4 text-accent-foreground" />
        )}
      </div>
      <div className={cn("flex-1 space-y-2", isUser && "flex flex-col items-end")}>
        <div
          className={cn(
            "rounded-lg px-4 py-3 leading-relaxed",
            isUser ? "bg-primary text-primary-foreground" : "bg-muted text-foreground",
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap text-pretty">{message.content}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>
        {!isUser && (
          <>
            {message.sources && message.sources.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">Sources</p>
                <div className="flex flex-wrap gap-2">
                  {message.sources.map((source) => (
                    <Card
                      key={source.id}
                      className="group max-w-xs border-border/50 transition-all hover:border-border hover:shadow-sm"
                    >
                      <CardContent className="p-3">
                        <div className="space-y-1.5">
                          <div className="flex items-start justify-between gap-2">
                            <h4 className="text-sm font-medium leading-tight text-foreground">{source.title}</h4>
                            {source.url && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-5 w-5 shrink-0 opacity-0 transition-opacity group-hover:opacity-100"
                                asChild
                              >
                                <a href={source.url} target="_blank" rel="noopener noreferrer">
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              </Button>
                            )}
                          </div>
                          <p className="line-clamp-2 text-xs text-muted-foreground">{source.snippet}</p>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </div>
            )}
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                className={cn("h-7 w-7", feedback === "up" && "bg-accent text-primary")}
                onClick={() => handleFeedback("up")}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={cn("h-7 w-7", feedback === "down" && "bg-accent text-destructive")}
                onClick={() => handleFeedback("down")}
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={cn("h-7 w-7", isSaved && "bg-accent text-primary")}
                onClick={handleSave}
              >
                <Bookmark className={cn("h-3.5 w-3.5", isSaved && "fill-current")} />
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

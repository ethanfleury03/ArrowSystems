"use client"

import { User, Sparkles, ThumbsUp, ThumbsDown, Bookmark, ExternalLink } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { useState } from "react"
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { submitFeedback, toggleSavedResponse } from "@/lib/api"
import type { Message } from "@/types/message"
import { useToast } from "@/hooks/use-toast"

interface ChatMessageProps {
  message: Message
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user"
  const [feedback, setFeedback] = useState<"up" | "down" | null>(null)
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState(false)
  const [isSaved, setIsSaved] = useState(message.metadata?.isSaved ?? false)
  const [isSaving, setIsSaving] = useState(false)
  const { toast } = useToast()

  const handleFeedback = async (type: "up" | "down") => {
    if (!message.metadata) {
      toast({
        variant: "destructive",
        title: "Feedback unavailable",
        description: "This response is missing metadata required to send feedback.",
      })
      return
    }

    if (feedback === type || isSubmittingFeedback) {
      return
    }

    const previousFeedback = feedback
    setFeedback(type)
    setIsSubmittingFeedback(true)

    try {
      await submitFeedback({
        query: message.metadata.query,
        answer: message.content,
        reasoning: message.metadata.reasoning,
        sources: message.metadata.structuredSources ?? [],
        document_sources: message.metadata.documentSources,
        confidence: message.metadata.confidence,
        intent_type: message.metadata.intentType,
        intent_confidence: message.metadata.intentConfidence,
        session_id: message.metadata.sessionId,
        matched_machine_name: message.metadata.matchedMachineName,
        top_k: message.metadata.topK ?? 10,
        alpha: message.metadata.alpha ?? 0.5,
        is_helpful: type === "up",
      })

      toast({
        title: type === "up" ? "Marked as helpful" : "Marked as unhelpful",
        description: "Thank you for your feedback!",
      })
    } catch (error) {
      setFeedback(previousFeedback)

      const description =
        error instanceof Error ? error.message : "Unable to submit feedback. Please try again."

      toast({
        variant: "destructive",
        title: "Feedback failed",
        description,
      })
    } finally {
      setIsSubmittingFeedback(false)
    }
  }

  const handleSave = async () => {
    if (!message.metadata) {
      toast({
        variant: "destructive",
        title: "Unable to save response",
        description: "This response is missing metadata required to save.",
      })
      return
    }

    if (isSaving) {
      return
    }

    const nextSavedState = !isSaved
    setIsSaved(nextSavedState)
    setIsSaving(true)

    try {
      const payload = {
        query: message.metadata.query,
        answer: message.content,
        reasoning: message.metadata.reasoning,
        sources: message.metadata.structuredSources ?? [],
        document_sources: message.metadata.documentSources,
        confidence: message.metadata.confidence,
        intent_type: message.metadata.intentType,
        intent_confidence: message.metadata.intentConfidence,
        session_id: message.metadata.sessionId,
        matched_machine_name: message.metadata.matchedMachineName,
        top_k: message.metadata.topK ?? 10,
        alpha: message.metadata.alpha ?? 0.5,
        is_saved: nextSavedState,
      }

      await toggleSavedResponse(payload)

      if (typeof window !== "undefined") {
        window.dispatchEvent(new CustomEvent("saved-responses:refresh"))
      }

      toast({
        title: nextSavedState ? "Response saved" : "Response removed",
        description: nextSavedState
          ? "You can find this response in the Saved tab."
          : "The response has been removed from your saved list.",
      })
    } catch (error) {
      setIsSaved(!nextSavedState)
      const description =
        error instanceof Error ? error.message : "Unable to update saved responses. Please try again."

      toast({
        variant: "destructive",
        title: nextSavedState ? "Failed to save response" : "Failed to unsave response",
        description,
      })
    } finally {
      setIsSaving(false)
    }
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
                disabled={isSubmittingFeedback}
                onClick={() => handleFeedback("up")}
              >
                <ThumbsUp className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={cn("h-7 w-7", feedback === "down" && "bg-accent text-destructive")}
                disabled={isSubmittingFeedback}
                onClick={() => handleFeedback("down")}
              >
                <ThumbsDown className="h-3.5 w-3.5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className={cn("h-7 w-7", isSaved ? "bg-accent text-amber-500" : "text-muted-foreground")}
                disabled={isSaving}
                onClick={handleSave}
              >
                <Bookmark
                  className={cn(
                    "h-3.5 w-3.5 transition-colors",
                    isSaved ? "fill-amber-400 text-amber-500" : "fill-background text-muted-foreground"
                  )}
                />
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

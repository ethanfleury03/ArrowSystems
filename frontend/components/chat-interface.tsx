"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ChatMessage } from "@/components/chat-message"
import { Sidebar } from "@/components/sidebar"
import { DocumentsPanel } from "@/components/documents-panel"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Send, Sparkles, Menu, MessageSquare, FileText } from "lucide-react"
import { sendQuery, getChatHistory, ChatHistoryItem } from "@/lib/api"
import type { Message, MessageSource } from "@/types/message"
import { QuerySettings } from "@/components/sidebar"

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const addNewConversationRef = useRef<((conversation: ChatHistoryItem) => void) | null>(null)
  const querySettingsRef = useRef<QuerySettings>({
    topK: 10,
    alpha: 0.5,
    dynamicWindowing: true,
  })

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    // Call RAG backend API
    try {
      const currentSettings = querySettingsRef.current
      const response = await sendQuery(userMessage.content, {
        top_k: currentSettings.topK,
        alpha: currentSettings.alpha,
        dynamic_windowing: currentSettings.dynamicWindowing,
      })
      const structuredSources = (response.sources ?? []).map((source) => ({
        id: source.id,
        name: source.name,
        pages: source.pages,
        content_type: source.content_type,
      }))
      const displaySources: MessageSource[] = (response.sources ?? []).map((source) => {
        const pagesLabel = source.pages && source.pages !== "N/A" ? `Pages: ${source.pages}` : "Pages not specified"
        const contentLabel = source.content_type ? source.content_type.toUpperCase() : "TEXT"
        return {
          id: source.id,
          title: source.name,
          snippet: `${contentLabel} · ${pagesLabel}`,
        }
      })

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        timestamp: new Date(),
        sources: displaySources,
        metadata: {
          query: userMessage.content,
          reasoning: response.reasoning,
          structuredSources,
          documentSources: response.document_sources || [],
          confidence: response.confidence,
          intentType: response.intent_type,
          intentConfidence: response.intent_confidence,
          sessionId: response.session_id ?? undefined,
          topK: currentSettings.topK,
          alpha: currentSettings.alpha,
          matchedMachineName: response.matched_machine_name ?? undefined,
          isSaved: response.is_saved ?? false,
        },
      }
      
      // Show summarization notice if query was summarized
      if (response.summarization_info?.was_summarized) {
        const info = response.summarization_info;
        const reduction = ((1 - info.summarized_length / info.original_length) * 100).toFixed(0);
        const contentType = info.content_type === 'email' ? 'email' : 
                           info.content_type === 'error' ? 'error log' : 'long question';
        
        // Add a system message showing summarization
        const summaryNotice: Message = {
          id: `summary-${Date.now()}`,
          role: "assistant",
          content: `📝 Note: Your ${contentType} was automatically summarized (${reduction}% shorter) to extract the key question.`,
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, summaryNotice])
      }
      
      setMessages((prev) => [...prev, assistantMessage])
      setIsLoading(false)

      // Add new conversation to history sidebar (auto-refresh)
      // Fetch the latest conversation from the database
      try {
        const historyResponse = await getChatHistory('api_user', 1)
        if (historyResponse.status === 'success' && historyResponse.history.length > 0) {
          const latestConversation = historyResponse.history[0]
          // Check if this is the conversation we just had (matching query text)
          if (latestConversation.query === userMessage.content && addNewConversationRef.current) {
            addNewConversationRef.current(latestConversation)
          }
        }
      } catch (error) {
        // Silently fail - this is just for auto-refresh, not critical
        console.debug('Failed to fetch latest conversation for history:', error)
      }
    } catch (error) {
      const errorText = error instanceof Error ? error.message : 'Failed to get response'
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Error: ${errorText}`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
      setIsLoading(false)

      // Don't save errors to history - only successful responses
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleNewConversationReady = (adder: (conversation: ChatHistoryItem) => void) => {
    addNewConversationRef.current = adder
  }

  const handleSettingsChange = (settings: QuerySettings) => {
    querySettingsRef.current = settings
  }

  const handleLoadConversation = (loadedMessages: Message[]) => {
    if (!loadedMessages?.length) {
      setSidebarOpen(false)
      return
    }

    setMessages((prev) => [...prev, ...loadedMessages])
    setSidebarOpen(false)
    setInput("")
    textareaRef.current?.focus()
  }

  return (
    <div className="flex h-screen">
      <Sidebar 
        isOpen={sidebarOpen} 
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onNewConversationReady={handleNewConversationReady}
        onSettingsChange={handleSettingsChange}
        onLoadConversation={handleLoadConversation}
      />

      <div className="flex flex-1 flex-col relative">
        {/* Header */}
        <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 relative z-10">
          <div className="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(!sidebarOpen)} className="md:hidden">
                <Menu className="h-5 w-5" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="hidden md:flex"
              >
                <Menu className="h-5 w-5" />
              </Button>
              <Sparkles className="h-5 w-5" />
              <h1 className="text-lg font-semibold">Arrow Systems Support</h1>
            </div>
          </div>
        </header>

        {/* Messages Container with Tabs */}
        <div className="flex-1 overflow-hidden relative z-10">
          {messages.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center gap-6 text-center pt-12">
              <div className="w-64 h-auto">
                <Image src="/asi-logo.png" alt="Arrow Systems Inc." width={256} height={256} className="w-full h-auto" />
              </div>
              <div className="space-y-2 mt-8">
                <h2 className="text-2xl font-semibold">How can I help you today?</h2>
                <p className="text-muted-foreground">Ask me anything about your knowledge base</p>
              </div>
            </div>
          ) : (
            <Tabs defaultValue="chat" className="flex h-full flex-col">
              <div className="border-b border-border px-4">
                <TabsList className="grid w-full max-w-4xl mx-auto grid-cols-2">
                  <TabsTrigger value="chat" className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    Chat
                  </TabsTrigger>
                  <TabsTrigger value="documents" className="flex items-center gap-2">
                    <FileText className="h-4 w-4" />
                    Documents
                    {(() => {
                      const lastAssistant = [...messages].reverse().find((msg) => msg.role === "assistant")
                      const docCount = lastAssistant?.metadata?.documentSources?.length || 0
                      return docCount > 0 ? (
                        <span className="ml-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs text-primary">
                          {docCount}
                        </span>
                      ) : null
                    })()}
                  </TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="chat" className="flex-1 overflow-y-auto m-0 mt-0">
                <div className="mx-auto max-w-4xl px-4 py-8">
                  <div className="space-y-6">
                    {messages.map((message) => (
                      <ChatMessage key={message.id} message={message} />
                    ))}
                    {isLoading && (
                      <div className="flex items-start gap-4">
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent">
                          <Sparkles className="h-4 w-4 text-accent-foreground" />
                        </div>
                        <div className="flex-1 space-y-2 pt-1">
                          <div className="flex gap-1">
                            <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]"></div>
                            <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]"></div>
                            <div className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground"></div>
                          </div>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="documents" className="flex-1 overflow-hidden m-0 mt-0">
                {(() => {
                  const lastAssistant = [...messages].reverse().find((msg) => msg.role === "assistant")
                  const documentSources = lastAssistant?.metadata?.documentSources || []
                  return <DocumentsPanel documentSources={documentSources} />
                })()}
              </TabsContent>
            </Tabs>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t border-border bg-background relative z-10">
          <div className="mx-auto max-w-4xl px-4 py-4">
            <form onSubmit={handleSubmit} className="relative">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Send a message..."
                className="min-h-[60px] resize-none pr-12 leading-relaxed"
                rows={1}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading}
                className="absolute bottom-2 right-2 h-8 w-8"
              >
                <Send className="h-4 w-4" />
                <span className="sr-only">Send message</span>
              </Button>
            </form>
            <p className="mt-2 text-center text-xs text-muted-foreground">
              AI can make mistakes. Verify important information.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

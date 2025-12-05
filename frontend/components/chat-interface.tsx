/**
 * ChatInterface Component
 * 
 * Main chat interface component that handles:
 * - Message display and input
 * - Query execution with configurable settings
 * - Query settings management (admin can customize, customers use hardcoded defaults)
 */
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
import { Send, Sparkles, Menu, MessageSquare, FileText, LogOut, AlertCircle } from "lucide-react"
import { sendQuery, getChatHistory, ChatHistoryItem, getCurrentUser, UserInfo } from "@/lib/api"
import type { Message, MessageSource } from "@/types/message"
import { QuerySettings } from "@/components/sidebar"
import { ErrorBoundary } from "@/components/ErrorBoundary"

/**
 * Get authentication token from browser cookies (client-side only)
 * The backend sets the JWT in a cookie (default: 'access_token', but may be 'auth_token')
 * Note: If cookie is httpOnly, this will return null and browser will send it automatically via credentials: "include"
 */
function getAuthToken(): string | null {
  if (typeof document === "undefined") return null;
  
  // Try both possible cookie names (backend default is 'access_token', but user mentioned 'auth_token')
  const cookies = document.cookie.split("; ");
  for (const cookieName of ["access_token", "auth_token"]) {
    const cookie = cookies.find((x) => x.startsWith(`${cookieName}=`));
    if (cookie) {
      return cookie.split("=")[1] ?? null;
    }
  }
  return null;
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [machineConfirmation, setMachineConfirmation] = useState(false)
  const [selectedMachine, setSelectedMachine] = useState<string | null>(null)
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null)
  const [ragEnabled, setRagEnabled] = useState<boolean | null>(null) // null = checking, true/false = known
  const [ragStatus, setRagStatus] = useState<'unknown' | 'ready' | 'warming' | 'disabled'>('unknown')
  const [ragLastError, setRagLastError] = useState<string | null>(null)
  const [checkingRag, setCheckingRag] = useState<boolean>(true)
  const ragPollingIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const onboardingShownRef = useRef(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const addNewConversationRef = useRef<((conversation: ChatHistoryItem) => void) | null>(null)
  
  // Default query settings - used for customers (hardcoded) and as fallback for admins
  const DEFAULT_QUERY_SETTINGS: QuerySettings = {
    topK: 18,  // Increased from 10 to 18 for better search coverage
    alpha: 0.5,
    dynamicWindowing: true,
  }
  
  const querySettingsRef = useRef<QuerySettings>(DEFAULT_QUERY_SETTINGS)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch user info and RAG status, show onboarding message on mount
  useEffect(() => {
    const fetchUserAndShowOnboarding = async () => {
      try {
        const user = await getCurrentUser()
        setUserInfo(user)
        
        // Check RAG status - use Next.js API route to avoid CORS preflight issues
        const checkRagStatus = async () => {
          try {
            // Call Next.js API route instead of backend directly
            // This avoids CORS preflight (OPTIONS) requests that fail with Cloud Run IAM auth
            const ragResponse = await fetch('/api/rag/status', {
              credentials: "include"
            })
            if (ragResponse.ok) {
              const ragData = await ragResponse.json()
              // Prefer rag_pipeline_initialized if present, fallback to initialized
              const initialized =
                ragData.rag_pipeline_initialized === true ||
                ragData.initialized === true
              const initializing = ragData.initializing === true
              const status = ragData.status || (initialized ? 'ready' : (initializing ? 'warming' : 'disabled'))
              
              setRagEnabled(initialized)
              setRagStatus(status as 'ready' | 'warming' | 'disabled')
              setRagLastError(ragData.last_error || null)
              setCheckingRag(false)
              
              // If warming, start polling
              if (status === 'warming' && !ragPollingIntervalRef.current) {
                startRagPolling()
              } else if (status === 'ready' && ragPollingIntervalRef.current) {
                // Stop polling when ready
                stopRagPolling()
              }
            } else if (ragResponse.status === 401 || ragResponse.status === 403) {
              // Authentication error - log but don't disable RAG (might be a temporary issue)
              console.warn("RAG status check returned auth error:", ragResponse.status)
              // Don't disable RAG on auth errors - backend might still be working
              setCheckingRag(false)
            } else {
              // Other errors - backend might be down or RAG truly disabled
              console.error("RAG status check failed:", ragResponse.status)
              setRagEnabled(false)
              setRagStatus('disabled')
              setCheckingRag(false)
            }
          } catch (error) {
            console.error("Failed to fetch RAG status:", error)
            // On network errors, don't assume RAG is disabled - might be temporary
            setCheckingRag(false)
          }
        }
        
        checkRagStatus()
        
        // Only show onboarding for customers and if not already shown and no messages
        if (user.role?.toUpperCase() === "CUSTOMER" && !onboardingShownRef.current && messages.length === 0) {
          const filteredMachines = user.machine_models && user.machine_models.length > 0
            ? user.machine_models.filter(m => m !== "GENERAL")
            : []
          
          const companyName = user.company_name || "Customer"
          
          // Format machine list as Markdown bullet points
          let machineListText = ""
          if (filteredMachines.length > 0) {
            // Use Markdown list syntax (- for bullet points)
            machineListText = filteredMachines.map(m => `- ${m}`).join("\n")
          } else {
            machineListText = "No machines assigned"
          }
          
          const onboardingMessage: Message = {
            id: `onboarding-${Date.now()}`,
            role: "assistant",
            content: `Hello ${companyName}, let's try to solve your problem.\n\nAccording to our records, you have the following machines:\n\n${machineListText}\n\nIs that correct?`,
            timestamp: new Date(),
          }
          
          setMessages([onboardingMessage])
          onboardingShownRef.current = true
        }
      } catch (error) {
        console.error("Failed to fetch user info:", error)
        // User is not authenticated, redirect to login
        window.location.href = "/login"
      }
    }
    
    fetchUserAndShowOnboarding()
    
    // Cleanup polling on unmount
    return () => {
      stopRagPolling()
    }
  }, [messages.length]) // Re-check when messages change (e.g., when loading a conversation)
  
  // Polling functions for RAG status
  const startRagPolling = () => {
    if (ragPollingIntervalRef.current) return // Already polling
    
    ragPollingIntervalRef.current = setInterval(async () => {
      try {
        // Call Next.js API route instead of backend directly
        // This avoids CORS preflight (OPTIONS) requests that fail with Cloud Run IAM auth
        const ragResponse = await fetch('/api/rag/status', {
          credentials: "include"
        })
        if (ragResponse.ok) {
          const ragData = await ragResponse.json()
          // Prefer rag_pipeline_initialized if present, fallback to initialized
          const initialized =
            ragData.rag_pipeline_initialized === true ||
            ragData.initialized === true
          const initializing = ragData.initializing === true
          const status = ragData.status || (initialized ? 'ready' : (initializing ? 'warming' : 'disabled'))
          
          setRagEnabled(initialized)
          setRagStatus(status as 'ready' | 'warming' | 'disabled')
          setRagLastError(ragData.last_error || null)
          
          // Stop polling when ready or disabled
          if (status === 'ready' || status === 'disabled') {
            stopRagPolling()
          }
        }
      } catch (error) {
        console.error("Failed to poll RAG status:", error)
      }
    }, 5000) // Poll every 5 seconds
  }
  
  const stopRagPolling = () => {
    if (ragPollingIntervalRef.current) {
      clearInterval(ragPollingIntervalRef.current)
      ragPollingIntervalRef.current = null
    }
  }

  const handleLogout = async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' })
      window.location.href = '/login'
    } catch (error) {
      console.error('Logout failed:', error)
      // Still redirect to login even if logout API fails
      window.location.href = '/login'
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return
    
    // Don't allow queries if RAG is not ready
    if (ragStatus !== 'ready') {
      let message = "Document search is currently unavailable. Please contact your administrator."
      if (ragStatus === 'warming') {
        message = "Document search is currently warming up. Please wait a moment and try again."
        // Start polling if not already polling
        if (!ragPollingIntervalRef.current) {
          startRagPolling()
        }
      } else if (ragStatus === 'disabled' && ragLastError) {
        message = `Document search is unavailable: ${ragLastError}. Please contact your administrator.`
      }
      
      const errorMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: message,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    const userInput = input.trim().toLowerCase()
    const originalInput = input.trim()
    setInput("")
    setIsLoading(true)

    // Handle machine confirmation for customers
    if (userInfo?.role?.toUpperCase() === "CUSTOMER" && !machineConfirmation) {
      // Add user message first
      setMessages((prev) => [...prev, userMessage])
      
      if (userInput === "yes" || userInput === "y") {
        // User confirmed machines
        const filteredMachines = userInfo.machine_models && userInfo.machine_models.length > 0
          ? userInfo.machine_models.filter(m => m !== "GENERAL")
          : []
        
        if (filteredMachines.length === 0) {
          // No machines assigned, just confirm
          const confirmMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "Perfect. What can I help you with today?",
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, confirmMessage])
          setMachineConfirmation(true)
          setIsLoading(false)
          return
        } else if (filteredMachines.length === 1) {
          // Only one machine, auto-select it
          const confirmMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `Perfect! I'll help you with your ${filteredMachines[0]}. What can I help you with today?`,
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, confirmMessage])
          setMachineConfirmation(true)
          setSelectedMachine(filteredMachines[0])
          setIsLoading(false)
          return
        } else {
          // Multiple machines - direct to sidebar selection
          const confirmMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: "Perfect! Select the machine you would like to query in the sidebar and go ahead and get started.",
            timestamp: new Date(),
          }
          setMessages((prev) => [...prev, confirmMessage])
          setMachineConfirmation(true)
          setIsLoading(false)
          return
        }
      } else if (userInput === "no" || userInput === "n") {
        // User said machines are wrong
        const rejectMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Understood. Your machine list can only be updated by your company's administrator.\n\nPlease contact your admin if something needs to be changed.",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, rejectMessage])
        setIsLoading(false)
        return
      } else {
        // User tried to ask a question before confirming
        const blockMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Please confirm your machines first by replying 'yes' or 'no'.",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, blockMessage])
        setIsLoading(false)
        return
      }
    }

    // Block queries until machine is selected (for customers with multiple machines)
    // Check if user has multiple machines and hasn't selected one yet
    if (userInfo?.role?.toUpperCase() === "CUSTOMER" && machineConfirmation) {
      const filteredMachines = userInfo.machine_models && userInfo.machine_models.length > 0
        ? userInfo.machine_models.filter(m => m !== "GENERAL")
        : []
      
      // If multiple machines and none selected, block queries and direct to sidebar
      if (filteredMachines.length > 1 && !selectedMachine) {
        setMessages((prev) => [...prev, userMessage])
        const blockMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Please select a machine from the sidebar first before asking questions.",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, blockMessage])
        setIsLoading(false)
        return
      }
    }

    // Add user message to chat (only if we're proceeding to RAG query)
    setMessages((prev) => [...prev, userMessage])

    // Call RAG backend API
    try {
      // Get effective query settings: use defaults for customers, allow customization for admins
      const effectiveSettings = getEffectiveQuerySettings(userInfo, querySettingsRef.current)
      const response = await sendQuery(userMessage.content, {
        top_k: effectiveSettings.topK,
        alpha: effectiveSettings.alpha,
        dynamic_windowing: effectiveSettings.dynamicWindowing,
        machine_confirmation: machineConfirmation || undefined,
        selected_machine: selectedMachine || undefined,
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
          topK: effectiveSettings.topK,
          alpha: effectiveSettings.alpha,
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
    } catch (error: any) {
      // Check if this is a RAG_WARMING error
      const errorData = error?.response?.data || error?.data || {}
      const ragCode = errorData?.code || errorData?.detail?.code
      
      if (ragCode === 'RAG_WARMING') {
        // Update state to warming and start polling
        setRagStatus('warming')
        setRagLastError(null)
        if (!ragPollingIntervalRef.current) {
          startRagPolling()
        }
        
        const warmingMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: "Document search is currently warming up. Please wait a moment and try again.",
          timestamp: new Date(),
        }
        setMessages((prev) => [...prev, warmingMessage])
        setIsLoading(false)
        return
      }
      
      const errorText = error instanceof Error ? error.message : (errorData?.detail || 'Failed to get response')
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

  /**
   * Get effective query settings based on user role.
   * Customers always use hardcoded defaults, admins can customize.
   */
  const getEffectiveQuerySettings = (currentUser: UserInfo | null, userSettings: QuerySettings): QuerySettings => {
    if (!currentUser || currentUser.role?.toUpperCase() !== "ADMIN") {
      // Customers always use defaults
      return DEFAULT_QUERY_SETTINGS
    }
    // Admins can customize
    return userSettings
  }

  const handleSettingsChange = (settings: QuerySettings) => {
    // Only allow settings changes for admins (customers won't see the UI anyway)
    if (userInfo?.role === 'ADMIN') {
      querySettingsRef.current = settings
    }
  }

  const handleLoadConversation = (loadedMessages: Message[]) => {
    if (!loadedMessages?.length) {
      setSidebarOpen(false)
      return
    }

    setMessages(loadedMessages)
    setSidebarOpen(false)
    setInput("")
    // Reset confirmation state when loading a conversation
    // Check if confirmation was already done in the loaded messages
    const hasConfirmation = loadedMessages.some(
      msg => msg.role === "assistant" && 
      (msg.content.includes("Perfect. What can I help you with today?") || 
       msg.content.includes("What can I help you with today?"))
    )
    setMachineConfirmation(hasConfirmation)
    textareaRef.current?.focus()
  }

  return (
    <div className="flex h-screen">
      <ErrorBoundary>
        <Sidebar 
          isOpen={sidebarOpen} 
          onToggle={() => setSidebarOpen(!sidebarOpen)}
          onNewConversationReady={handleNewConversationReady}
        onSettingsChange={handleSettingsChange}
        onLoadConversation={handleLoadConversation}
        selectedMachine={selectedMachine}
        onMachineChange={setSelectedMachine}
        userInfo={userInfo}
      />
      </ErrorBoundary>

      <ErrorBoundary>
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
            <div className="flex items-center gap-2">
              {userInfo && (
                <span className="text-sm text-muted-foreground hidden sm:inline">
                  {userInfo.email}
                </span>
              )}
              <Button variant="ghost" size="icon" onClick={handleLogout} title="Logout">
                <LogOut className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </header>

        {/* RAG Status Banner */}
        {ragStatus === 'warming' && (
          <div className="bg-yellow-500/10 border-b border-yellow-500/20 px-4 py-3">
            <div className="mx-auto max-w-4xl flex items-center gap-2 text-sm text-yellow-700 dark:text-yellow-400">
              <AlertCircle className="h-4 w-4 flex-shrink-0 animate-pulse" />
              <span>Preparing document search. This may take ~20–40 seconds on first use.</span>
            </div>
          </div>
        )}
        {ragStatus === 'disabled' && (
          <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-3">
            <div className="mx-auto max-w-4xl flex items-center gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 flex-shrink-0" />
              <span>
                {ragLastError 
                  ? `Document search is unavailable: ${ragLastError}. Please contact your administrator.`
                  : 'Document search is currently unavailable because the RAG index is not loaded. Please contact your administrator.'}
              </span>
            </div>
          </div>
        )}

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
                placeholder={ragStatus !== 'ready' ? "Document search is not ready..." : "Send a message..."}
                className="min-h-[60px] resize-none pr-12 leading-relaxed"
                rows={1}
                disabled={ragStatus !== 'ready'}
              />
              <Button
                type="submit"
                size="icon"
                disabled={!input.trim() || isLoading || ragStatus !== 'ready'}
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
      </ErrorBoundary>
    </div>
  )
}

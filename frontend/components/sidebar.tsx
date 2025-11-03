"use client"

import { Settings, FileText, History, Menu, LogOut, User, Bookmark, Search, MessageSquare, Clock, Download, Trash2, Database, Server, CheckCircle2, XCircle, Info } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { useState, useEffect } from "react"
import { getChatHistory, ChatHistoryItem, getHealth, getSavedResponses, SavedResponse } from "@/lib/api"

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
  onNewConversationReady?: (adder: (conversation: ChatHistoryItem) => void) => void
  onSettingsChange?: (settings: QuerySettings) => void
}

export interface QuerySettings {
  topK: number
  alpha: number
  dynamicWindowing: boolean
}

// Removed mock data - will use real data from API

export function Sidebar({ isOpen, onToggle, onNewConversationReady, onSettingsChange }: SidebarProps) {
  const [activeTab, setActiveTab] = useState<"options" | "saved" | "history" | "settings">("options")
  const [searchQuery, setSearchQuery] = useState("")
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([])
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)
  const [showChatHistory, setShowChatHistory] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [savedResponses, setSavedResponses] = useState<SavedResponse[]>([])
  const [isLoadingSaved, setIsLoadingSaved] = useState(false)
  
  // Settings state
  const [querySettings, setQuerySettings] = useState<QuerySettings>({
    topK: 10,
    alpha: 0.5,
    dynamicWindowing: true,
  })
  
  // System info state
  const [backendStatus, setBackendStatus] = useState<boolean | null>(null)
  const [databaseStatus, setDatabaseStatus] = useState<boolean | null>(null)

  // Expose function to add new conversation to parent (ChatInterface)
  useEffect(() => {
    if (onNewConversationReady) {
      // Create a function that adds conversation to local state
      const addNewConversation = (conversation: ChatHistoryItem) => {
        setChatHistory((prev) => {
          // Check if conversation already exists (avoid duplicates)
          const exists = prev.some((item) => item.id === conversation.id)
          if (exists) {
            return prev
          }
          // Add to top of list
          return [conversation, ...prev]
        })
      }
      
      // Call the callback with our local function
      // This allows ChatInterface to store and call it when needed
      onNewConversationReady(addNewConversation)
    }
  }, [onNewConversationReady])

  // Fetch chat history when showing history tab
  useEffect(() => {
    if (showChatHistory && chatHistory.length === 0 && !isLoadingHistory) {
      fetchChatHistory()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showChatHistory])

  // Fetch saved responses when showing saved tab
  useEffect(() => {
    if (activeTab === "saved" && savedResponses.length === 0 && !isLoadingSaved) {
      fetchSavedResponses()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab])

  const fetchChatHistory = async () => {
    setIsLoadingHistory(true)
    try {
      const response = await getChatHistory('api_user', 50)
      if (response.status === 'success') {
        setChatHistory(response.history)
      }
    } catch (error) {
      console.error('Failed to fetch chat history:', error)
      setChatHistory([])
    } finally {
      setIsLoadingHistory(false)
    }
  }

  const fetchSavedResponses = async () => {
    setIsLoadingSaved(true)
    try {
      const response = await getSavedResponses(50, 1)  // Changed to 1 - show responses with at least 1 thumbs up
      if (response.status === 'success') {
        setSavedResponses(response.saved)
      }
    } catch (error) {
      console.error('Failed to fetch saved responses:', error)
      setSavedResponses([])
    } finally {
      setIsLoadingSaved(false)
    }
  }

  const filteredChatHistory = chatHistory.filter((chat) => 
    chat.query.toLowerCase().includes(searchQuery.toLowerCase()) ||
    chat.answer.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleChatHistoryClick = () => {
    setShowChatHistory(true)
    setActiveTab("history")
    if (chatHistory.length === 0) {
      fetchChatHistory()
    }
  }

  const handleBackToOptions = () => {
    setShowChatHistory(false)
    setActiveTab("options")
    setSearchQuery("")
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp)
      const now = new Date()
      const diffMs = now.getTime() - date.getTime()
      const diffMins = Math.floor(diffMs / 60000)
      const diffHours = Math.floor(diffMs / 3600000)
      const diffDays = Math.floor(diffMs / 86400000)

      if (diffMins < 1) return "Just now"
      if (diffMins < 60) return `${diffMins}m ago`
      if (diffHours < 24) return `${diffHours}h ago`
      if (diffDays < 7) return `${diffDays}d ago`
      
      return date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
      })
    } catch {
      return timestamp
    }
  }

  const handleLogout = () => {
    // TODO: Implement actual logout logic
    console.log("Logging out...")
  }

  const handleSettingsClick = () => {
    setShowSettings(true)
    setActiveTab("settings")
    checkSystemStatus()
  }

  const handleBackFromSettings = () => {
    setShowSettings(false)
    setActiveTab("options")
  }

  const updateQuerySettings = (updates: Partial<QuerySettings>) => {
    const newSettings = { ...querySettings, ...updates }
    setQuerySettings(newSettings)
    if (onSettingsChange) {
      onSettingsChange(newSettings)
    }
    // Save to localStorage for persistence
    localStorage.setItem('querySettings', JSON.stringify(newSettings))
  }

  // Load settings from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('querySettings')
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        setQuerySettings(parsed)
        if (onSettingsChange) {
          onSettingsChange(parsed)
        }
      } catch (e) {
        console.error('Failed to load settings:', e)
        if (onSettingsChange) {
          onSettingsChange(querySettings)
        }
      }
    } else if (onSettingsChange) {
      onSettingsChange(querySettings)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const checkSystemStatus = async () => {
    try {
      const health = await getHealth()
      setBackendStatus(health)
      // Database status would come from a separate endpoint
      // For now, assume it's available if backend is healthy
      setDatabaseStatus(health)
    } catch {
      setBackendStatus(false)
      setDatabaseStatus(false)
    }
  }

  const handleClearHistory = async () => {
    if (!confirm('Are you sure you want to clear all chat history? This cannot be undone.')) {
      return
    }
    // TODO: Implement API endpoint to clear history
    setChatHistory([])
    alert('Chat history cleared')
  }

  const handleExportHistory = () => {
    if (chatHistory.length === 0) {
      alert('No history to export')
      return
    }
    
    const dataStr = JSON.stringify(chatHistory, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `chat-history-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && <div className="fixed inset-0 z-40 bg-background/80 backdrop-blur-sm md:hidden" onClick={onToggle} />}

      {/* Sidebar */}
      <aside
        className={`fixed left-0 top-0 z-50 h-screen w-64 border-r border-border bg-background transition-transform duration-300 md:relative md:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Sidebar Header */}
          <div className="flex h-14 items-center justify-between border-b border-border px-4">
            <h2 className="font-semibold">Options</h2>
            <Button variant="ghost" size="icon" onClick={onToggle} className="md:hidden">
              <Menu className="h-5 w-5" />
            </Button>
          </div>

          <div className="border-b border-border p-4">
            <div className="flex items-center gap-3 rounded-lg bg-muted/50 p-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10">
                <User className="h-5 w-5 text-primary" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">John Doe</p>
                <p className="text-xs text-muted-foreground">john@example.com</p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="mt-2 w-full justify-start gap-2 text-muted-foreground hover:text-foreground"
              onClick={handleLogout}
            >
              <LogOut className="h-4 w-4" />
              Log out
            </Button>
          </div>

          <div className="border-b border-border p-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                type="text"
                placeholder="Search chat history..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
          </div>

          <div className="flex border-b border-border">
            <button
              onClick={() => {
                setActiveTab("options")
                setShowChatHistory(false)
                setShowSettings(false)
              }}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === "options" && !showChatHistory && !showSettings
                  ? "border-b-2 border-primary text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Options
            </button>
            <button
              onClick={() => {
                setActiveTab("saved")
                setShowChatHistory(false)
                setShowSettings(false)
              }}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === "saved" && !showChatHistory && !showSettings
                  ? "border-b-2 border-primary text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Saved
            </button>
          </div>

          {showChatHistory ? (
            <div className="flex-1 overflow-y-auto flex flex-col">
              {/* Chat History Header */}
              <div className="border-b border-border p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <History className="h-4 w-4 text-primary" />
                    <h3 className="font-semibold text-sm">Chat History</h3>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleBackToOptions}
                    className="h-7 px-2"
                  >
                    Back
                  </Button>
                </div>
                {isLoadingHistory && (
                  <p className="text-xs text-muted-foreground">Loading history...</p>
                )}
                {!isLoadingHistory && chatHistory.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {filteredChatHistory.length} {filteredChatHistory.length === 1 ? "conversation" : "conversations"}
                  </p>
                )}
              </div>

              {/* Chat History List */}
              <div className="flex-1 overflow-y-auto p-4">
                {isLoadingHistory ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-sm text-muted-foreground">Loading...</div>
                  </div>
                ) : filteredChatHistory.length > 0 ? (
                  <div className="space-y-2">
                    {filteredChatHistory.map((chat) => (
                      <button
                        key={chat.id}
                        className="w-full rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-accent hover:border-primary/50 group"
                        onClick={() => {
                          // TODO: Load this conversation into the chat
                          console.log("Load conversation:", chat.id)
                        }}
                      >
                        <div className="flex items-start gap-2">
                          <MessageSquare className="h-4 w-4 shrink-0 text-muted-foreground mt-0.5 group-hover:text-primary transition-colors" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-foreground line-clamp-1 leading-relaxed">
                              {chat.query}
                            </p>
                            <p className="text-xs text-muted-foreground line-clamp-2 mt-1 leading-relaxed">
                              {chat.answer}
                            </p>
                            <div className="flex items-center gap-3 mt-2">
                              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                <Clock className="h-3 w-3" />
                                {formatTimestamp(chat.timestamp)}
                              </div>
                              {chat.intent_type && (
                                <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
                                  {chat.intent_type}
                                </span>
                              )}
                              {chat.confidence !== undefined && (
                                <span className="text-xs text-muted-foreground">
                                  {(chat.confidence * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <History className="h-12 w-12 text-muted-foreground/30 mb-3" />
                    <p className="text-sm font-medium text-foreground mb-1">No chat history</p>
                    <p className="text-xs text-muted-foreground">
                      {searchQuery ? "No conversations match your search" : "Start a conversation to see your history here"}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ) : activeTab === "options" ? (
            <div className="flex-1 overflow-y-auto">
              {searchQuery && chatHistory.length > 0 ? (
                <div className="p-4">
                  <p className="mb-3 text-xs font-medium text-muted-foreground">
                    {filteredChatHistory.length} {filteredChatHistory.length === 1 ? "result" : "results"}
                  </p>
                  <div className="space-y-2">
                    {filteredChatHistory.length > 0 ? (
                      filteredChatHistory.map((chat) => (
                        <button
                          key={chat.id}
                          className="w-full rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-accent"
                          onClick={() => {
                            setShowChatHistory(true)
                            setActiveTab("history")
                          }}
                        >
                          <p className="text-sm font-medium text-foreground line-clamp-1">{chat.query}</p>
                          <p className="mt-1 text-xs text-muted-foreground line-clamp-1">{chat.answer}</p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {formatTimestamp(chat.timestamp)}
                          </p>
                        </button>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground">No chats found</p>
                    )}
                  </div>
                </div>
              ) : (
                <nav className="space-y-1 p-4">
                  <Button 
                    variant="ghost" 
                    className="w-full justify-start gap-3"
                    onClick={handleChatHistoryClick}
                  >
                    <History className="h-4 w-4" />
                    Chat History
                  </Button>
                  <Button variant="ghost" className="w-full justify-start gap-3">
                    <FileText className="h-4 w-4" />
                    Documents
                  </Button>
                  <Button 
                    variant="ghost" 
                    className="w-full justify-start gap-3"
                    onClick={handleSettingsClick}
                  >
                    <Settings className="h-4 w-4" />
                    Settings
                  </Button>
                </nav>
              )}
            </div>
          ) : showSettings ? (
            <div className="flex-1 overflow-y-auto flex flex-col">
              {/* Settings Header */}
              <div className="border-b border-border p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Settings className="h-4 w-4 text-primary" />
                    <h3 className="font-semibold text-sm">Settings</h3>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleBackFromSettings}
                    className="h-7 px-2"
                  >
                    Back
                  </Button>
                </div>
              </div>

              {/* Settings Content */}
              <div className="flex-1 overflow-y-auto p-4 space-y-6">
                {/* Query Preferences */}
                <div className="space-y-4">
                  <h4 className="text-sm font-semibold text-foreground">Query Preferences</h4>
                  
                  {/* Top-K Slider */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="top-k" className="text-xs">Chunks to Retrieve (Top-K)</Label>
                      <span className="text-xs text-muted-foreground">{querySettings.topK}</span>
                    </div>
                    <Slider
                      id="top-k"
                      min={1}
                      max={50}
                      step={1}
                      value={[querySettings.topK]}
                      onValueChange={(value) => updateQuerySettings({ topK: value[0] })}
                    />
                    <p className="text-xs text-muted-foreground">
                      More chunks = more context but slower responses
                    </p>
                  </div>

                  {/* Alpha Slider */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="alpha" className="text-xs">Search Mode (Alpha)</Label>
                      <span className="text-xs text-muted-foreground">{querySettings.alpha.toFixed(1)}</span>
                    </div>
                    <Slider
                      id="alpha"
                      min={0}
                      max={1}
                      step={0.1}
                      value={[querySettings.alpha]}
                      onValueChange={(value) => updateQuerySettings({ alpha: value[0] })}
                    />
                    <div className="flex items-center gap-2 text-xs">
                      <span className={querySettings.alpha < 0.3 ? "font-semibold text-foreground" : "text-muted-foreground"}>Keyword</span>
                      <div className="flex-1 h-1 bg-muted rounded-full">
                        <div 
                          className="h-full bg-primary rounded-full transition-all"
                          style={{ width: `${querySettings.alpha * 100}%` }}
                        />
                      </div>
                      <span className={querySettings.alpha > 0.7 ? "font-semibold text-foreground" : "text-muted-foreground"}>Semantic</span>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {querySettings.alpha < 0.3 ? "🔤 Keyword-focused" : querySettings.alpha > 0.7 ? "🧠 Semantic-focused" : "⚖️ Balanced (recommended)"}
                    </p>
                  </div>

                  {/* Dynamic Windowing Toggle */}
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="dynamic-windowing" className="text-xs">Dynamic Windowing</Label>
                      <p className="text-xs text-muted-foreground">Optimize context window automatically</p>
                    </div>
                    <Switch
                      id="dynamic-windowing"
                      checked={querySettings.dynamicWindowing}
                      onCheckedChange={(checked) => updateQuerySettings({ dynamicWindowing: checked })}
                    />
                  </div>
                </div>

                <div className="border-t border-border" />

                {/* Data Management */}
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-foreground">Data Management</h4>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full justify-start gap-2"
                    onClick={handleExportHistory}
                  >
                    <Download className="h-4 w-4" />
                    Export Chat History
                  </Button>

                  <Button
                    variant="destructive"
                    size="sm"
                    className="w-full justify-start gap-2"
                    onClick={handleClearHistory}
                  >
                    <Trash2 className="h-4 w-4" />
                    Clear Chat History
                  </Button>
                </div>

                <div className="border-t border-border" />

                {/* System Information */}
                <div className="space-y-3">
                  <h4 className="text-sm font-semibold text-foreground">System Information</h4>
                  
                  <div className="space-y-2 text-xs">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Server className="h-3 w-3 text-muted-foreground" />
                        <span className="text-muted-foreground">Backend API</span>
                      </div>
                      {backendStatus === null ? (
                        <Info className="h-3 w-3 text-muted-foreground" />
                      ) : backendStatus ? (
                        <div className="flex items-center gap-1 text-green-600">
                          <CheckCircle2 className="h-3 w-3" />
                          <span>Online</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-red-600">
                          <XCircle className="h-3 w-3" />
                          <span>Offline</span>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Database className="h-3 w-3 text-muted-foreground" />
                        <span className="text-muted-foreground">Database</span>
                      </div>
                      {databaseStatus === null ? (
                        <Info className="h-3 w-3 text-muted-foreground" />
                      ) : databaseStatus ? (
                        <div className="flex items-center gap-1 text-green-600">
                          <CheckCircle2 className="h-3 w-3" />
                          <span>Connected</span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1 text-red-600">
                          <XCircle className="h-3 w-3" />
                          <span>Disconnected</span>
                        </div>
                      )}
                    </div>

                    <div className="flex items-center justify-between pt-1">
                      <span className="text-muted-foreground">Version</span>
                      <span className="text-foreground">1.0.0</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto flex flex-col">
              {/* Saved Responses Header */}
              <div className="border-b border-border p-4">
                <div className="flex items-center gap-2 mb-3">
                  <Bookmark className="h-4 w-4 text-primary" />
                  <h3 className="font-semibold text-sm">Saved Responses</h3>
                </div>
                {isLoadingSaved && (
                  <p className="text-xs text-muted-foreground">Loading saved responses...</p>
                )}
                {!isLoadingSaved && savedResponses.length > 0 && (
                  <p className="text-xs text-muted-foreground">
                    {savedResponses.length} {savedResponses.length === 1 ? "saved response" : "saved responses"}
                  </p>
                )}
              </div>

              {/* Saved Responses List */}
              <div className="flex-1 overflow-y-auto p-4">
                {isLoadingSaved ? (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-sm text-muted-foreground">Loading...</div>
                  </div>
                ) : savedResponses.length > 0 ? (
                  <div className="space-y-2">
                    {savedResponses.map((response) => (
                      <button
                        key={response.id}
                        className="w-full rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-accent hover:border-primary/50 group"
                        onClick={() => {
                          // TODO: Load this saved response into the chat
                          console.log("Load saved response:", response.id)
                        }}
                      >
                        <div className="flex items-start gap-2">
                          <Bookmark className="h-4 w-4 shrink-0 text-primary mt-0.5 fill-current" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-foreground line-clamp-1 leading-relaxed">
                              {response.query}
                            </p>
                            <p className="text-xs text-muted-foreground line-clamp-2 mt-1 leading-relaxed">
                              {response.answer}
                            </p>
                            <div className="flex items-center gap-3 mt-2">
                              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                <span>👍 {response.helpful_count}</span>
                                {response.unhelpful_count > 0 && (
                                  <span>👎 {response.unhelpful_count}</span>
                                )}
                              </div>
                              {response.sources && response.sources.length > 0 && (
                                <span className="text-xs text-muted-foreground">
                                  {response.sources.length} {response.sources.length === 1 ? "source" : "sources"}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <Bookmark className="h-12 w-12 text-muted-foreground/30 mb-3" />
                    <p className="text-sm font-medium text-foreground mb-1">No saved responses</p>
                    <p className="text-xs text-muted-foreground">
                      Responses marked as helpful will appear here
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Sidebar Footer */}
          <div className="border-t border-border p-4">
            <p className="text-xs text-muted-foreground">RAG Assistant v1.0</p>
          </div>
        </div>
      </aside>
    </>
  )
}

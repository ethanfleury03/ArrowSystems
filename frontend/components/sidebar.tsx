"use client"

import { Settings, FileText, History, Menu, LogOut, User, Bookmark, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { useState } from "react"

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
}

const chatHistory = [
  {
    id: "1",
    title: "Introduction to Machine Learning",
    date: new Date("2024-01-20"),
  },
  {
    id: "2",
    title: "React Best Practices 2024",
    date: new Date("2024-01-19"),
  },
  {
    id: "3",
    title: "Database Optimization Techniques",
    date: new Date("2024-01-18"),
  },
  {
    id: "4",
    title: "Understanding Neural Networks",
    date: new Date("2024-01-17"),
  },
  {
    id: "5",
    title: "API Design Principles",
    date: new Date("2024-01-16"),
  },
]

const savedResponses = [
  {
    id: "1",
    snippet: "The capital of France is Paris, which is located in the north-central part...",
    date: new Date("2024-01-15"),
  },
  {
    id: "2",
    snippet: "Machine learning is a subset of artificial intelligence that enables...",
    date: new Date("2024-01-14"),
  },
  {
    id: "3",
    snippet: "React is a JavaScript library for building user interfaces, developed by...",
    date: new Date("2024-01-13"),
  },
]

export function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const [activeTab, setActiveTab] = useState<"options" | "saved">("options")
  const [searchQuery, setSearchQuery] = useState("")

  const filteredChatHistory = chatHistory.filter((chat) => chat.title.toLowerCase().includes(searchQuery.toLowerCase()))

  const handleLogout = () => {
    // TODO: Implement actual logout logic
    console.log("Logging out...")
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
              onClick={() => setActiveTab("options")}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === "options"
                  ? "border-b-2 border-primary text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Options
            </button>
            <button
              onClick={() => setActiveTab("saved")}
              className={`flex-1 px-4 py-2 text-sm font-medium transition-colors ${
                activeTab === "saved"
                  ? "border-b-2 border-primary text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Saved
            </button>
          </div>

          {activeTab === "options" ? (
            <div className="flex-1 overflow-y-auto">
              {searchQuery ? (
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
                        >
                          <p className="text-sm font-medium text-foreground">{chat.title}</p>
                          <p className="mt-1 text-xs text-muted-foreground">
                            {chat.date.toLocaleDateString("en-US", {
                              month: "short",
                              day: "numeric",
                              year: "numeric",
                            })}
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
                  <Button variant="ghost" className="w-full justify-start gap-3">
                    <History className="h-4 w-4" />
                    Chat History
                  </Button>
                  <Button variant="ghost" className="w-full justify-start gap-3">
                    <FileText className="h-4 w-4" />
                    Documents
                  </Button>
                  <Button variant="ghost" className="w-full justify-start gap-3">
                    <Settings className="h-4 w-4" />
                    Settings
                  </Button>
                </nav>
              )}
            </div>
          ) : (
            <div className="flex-1 overflow-y-auto p-4">
              <div className="space-y-2">
                {savedResponses.map((response) => (
                  <button
                    key={response.id}
                    className="w-full rounded-lg border border-border bg-card p-3 text-left transition-colors hover:bg-accent"
                  >
                    <div className="flex items-start gap-2">
                      <Bookmark className="h-4 w-4 shrink-0 text-primary mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-foreground line-clamp-2 leading-relaxed">{response.snippet}</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {response.date.toLocaleDateString("en-US", {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                          })}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
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

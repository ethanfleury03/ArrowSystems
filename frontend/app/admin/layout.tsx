"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Toaster } from "@/components/ui/toaster";
import type { ReactNode } from "react";
import { Menu, Users, FileText, Activity, ArrowLeft, X, BarChart3, Settings, Cog, LogOut, Search, Ticket } from "lucide-react";
import { getCurrentUser } from "@/lib/api";

interface AdminLayoutProps {
  children: ReactNode;
}

interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { label: "User Management", href: "/admin/users", icon: Users },
  { label: "Documents", href: "/admin/documents", icon: FileText },
  { label: "Tickets", href: "/admin/tickets", icon: Ticket },
  { label: "Query Insights", href: "/admin/query-insights", icon: Search },
  { label: "Analytics", href: "/admin/analytics", icon: BarChart3 },
  { label: "Settings", href: "/admin/settings", icon: Settings },
];

export default function AdminLayout({ children }: AdminLayoutProps) {
  const router = useRouter();
  const pathname = usePathname();
  const { toast } = useToast();

  // Cache admin check in sessionStorage to avoid re-checking on every navigation
  // Always start with null to ensure server/client hydration match
  const [isAdmin, setIsAdmin] = useState<boolean | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [userProfile, setUserProfile] = useState<{ email?: string | null; name?: string | null; role?: string | null } | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  // Set mounted state after hydration
  useEffect(() => {
    setIsMounted(true);
    // Check sessionStorage after mount to avoid hydration mismatch
    // Only trust 'true' cache, always re-check if 'false' or missing
    if (typeof window !== 'undefined') {
      const cached = sessionStorage.getItem('admin_check');
      if (cached === 'true') {
        setIsAdmin(true);
        return; // Skip API check if cached as admin
      }
      // Don't cache 'false' - always re-check to allow retry
    }
  }, [router]);

  // Validate token and ensure admin access (only once per session)
  useEffect(() => {
    // Wait for mount before checking
    if (!isMounted) return;
    
    // If already checked and cached as admin, skip the check
    if (isAdmin === true && typeof window !== 'undefined' && sessionStorage.getItem('admin_check') === 'true') {
      return;
    }

    let isMountedRef = true;
    let hasRedirected = false;

    const checkAdminAccess = async () => {
      try {
        // Call /api/auth/me to get user from cookie-based JWT
        const user = await getCurrentUser();

        if (!isMountedRef || hasRedirected) return;

        // Verify user has ADMIN role
        if (user.role !== "ADMIN") {
          // Don't cache 'false' - allow retry on next visit
          hasRedirected = true;
          setIsAdmin(false);
          // Use window.location to prevent loop
          window.location.href = "/";
          return;
        }

        // User is admin, set state and cache
        if (!isMountedRef || hasRedirected) return;
        
        if (typeof window !== 'undefined') {
          sessionStorage.setItem('admin_check', 'true');
        }
        setIsAdmin(true);
        setUserProfile({
          email: user.email,
          name: user.name || null,
          role: user.role,
        });
      } catch (error) {
        if (!isMountedRef || hasRedirected) return;

        // Don't cache 'false' - allow retry on next visit
        hasRedirected = true;
        console.error("Admin access check failed:", error);
        setIsAdmin(false);
        // Use window.location to prevent loop
        window.location.href = "/";
      }
    };

    checkAdminAccess();

    return () => {
      isMountedRef = false;
    };
  }, [router, toast, isAdmin, isMounted]);

  const displayName = useMemo(() => {
    if (userProfile?.name && userProfile.name.trim().length > 0) {
      return userProfile.name;
    }
    if (userProfile?.email && userProfile.email.includes("@")) {
      return userProfile.email.split("@")[0];
    }
    return "Administrator";
  }, [userProfile]);

  const displayEmail = useMemo(() => {
    if (userProfile?.email && userProfile.email.trim().length > 0) {
      return userProfile.email;
    }
    if (userProfile?.name && userProfile.name.trim().length > 0) {
      return `${userProfile.name.replace(/\s+/g, "").toLowerCase()}@example.com`;
    }
    return "admin@example.com";
  }, [userProfile]);

  const toggleSidebar = () => setIsSidebarOpen((prev) => !prev);

  const closeSidebar = () => setIsSidebarOpen(false);

  const isRouteActive = (href: string) => pathname.startsWith(href);

  const handleLogout = async () => {
    try {
      await fetch('/api/auth/logout', { method: 'POST' });
      window.location.href = '/login';
    } catch (error) {
      console.error('Logout failed:', error);
      // Still redirect to login even if logout API fails
      window.location.href = '/login';
    }
  };

  // Show loading state during initial check (consistent server/client render)
  if (isAdmin === null) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-muted/40">
        <p className="text-muted-foreground">Validating access...</p>
      </div>
    );
  }

  // If not admin, show nothing (will redirect)
  if (!isAdmin) {
    return null;
  }

  return (
    <div className="flex min-h-screen bg-muted/40">
      {/* Sidebar for desktop */}
      <aside className="hidden w-64 border-r border-border bg-background md:flex md:flex-col">
        <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Users className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{displayName}</p>
              <p className="truncate text-xs text-muted-foreground">{displayEmail}</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={() => router.push("/")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
        </div>

        <nav className="flex-1 space-y-1 px-3 py-4">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = isRouteActive(href);
            const isSettings = href === "/admin/settings";
            return (
              <div key={href}>
                <Link
                  href={href}
                  className={`group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    active
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4 flex-shrink-0" />
                  <span>{label}</span>
                </Link>
                {isSettings && active && (
                  <div className="ml-7 mt-1 space-y-1">
                    <Link
                      href="/admin/settings/machine-models"
                      className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        pathname === "/admin/settings/machine-models"
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                      }`}
                    >
                      <Cog className="h-3 w-3" />
                      <span>Machine Models</span>
                    </Link>
                    <Link
                      href="/admin/settings/ticket-index"
                      className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        pathname === "/admin/settings/ticket-index"
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                      }`}
                    >
                      <Ticket className="h-3 w-3" />
                      <span>Ticket Index</span>
                    </Link>
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </aside>

      {/* Mobile sidebar */}
      <div
        className={`fixed inset-0 z-40 bg-background/80 backdrop-blur-sm transition-opacity md:hidden ${
          isSidebarOpen ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={closeSidebar}
      />
      <aside
        className={`fixed top-0 left-0 z-50 h-full w-64 border-r border-border bg-background transition-transform md:hidden ${
          isSidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary/10 text-primary">
              <Menu className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold">{displayName}</p>
              <p className="truncate text-xs text-muted-foreground">{displayEmail}</p>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={closeSidebar}>
            <X className="h-4 w-4" />
          </Button>
        </div>
        <nav className="space-y-1 px-3 py-4">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = isRouteActive(href);
            const isSettings = href === "/admin/settings";
            return (
              <div key={href}>
                <Link
                  href={href}
                  onClick={closeSidebar}
                  className={`group flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors ${
                    active
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                  }`}
                >
                  <Icon className="h-4 w-4 flex-shrink-0" />
                  <span>{label}</span>
                </Link>
                {isSettings && active && (
                  <div className="ml-7 mt-1 space-y-1">
                    <Link
                      href="/admin/settings/machine-models"
                      onClick={closeSidebar}
                      className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        pathname === "/admin/settings/machine-models"
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                      }`}
                    >
                      <Cog className="h-3 w-3" />
                      <span>Machine Models</span>
                    </Link>
                    <Link
                      href="/admin/settings/ticket-index"
                      onClick={closeSidebar}
                      className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        pathname === "/admin/settings/ticket-index"
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted/70 hover:text-foreground"
                      }`}
                    >
                      <Ticket className="h-3 w-3" />
                      <span>Ticket Index</span>
                    </Link>
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </aside>

      {/* Main content */}
      <div className="flex-1 md:pl-0">
        <header className="sticky top-0 z-30 flex items-center justify-between border-b border-border bg-background px-4 py-3 md:px-6">
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="md:hidden" onClick={toggleSidebar}>
              <Menu className="h-5 w-5" />
            </Button>
            <Button
              variant="ghost"
              className="flex items-center gap-2 md:hidden"
              onClick={() => router.push("/")}
            >
              <ArrowLeft className="h-4 w-4" />
              <span>Back to Home</span>
            </Button>
          </div>
          <div className="flex items-center gap-2">
            {displayEmail && (
              <span className="text-sm text-muted-foreground hidden sm:inline">
                {displayEmail}
              </span>
            )}
            <Button variant="ghost" size="icon" onClick={handleLogout} title="Logout">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </header>

        <main className="px-4 py-6 md:px-8">
          <div className="mx-auto w-full max-w-[calc(100vw-16rem)] md:max-w-[calc(100vw-16rem)]">
            {children}
          </div>
        </main>
      </div>
      <Toaster />
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface InviteAcceptFormProps {
  initialToken: string;
}

interface InviteValidateResponse {
  email: string;
  name?: string | null;
}

export function InviteAcceptForm({ initialToken }: InviteAcceptFormProps) {
  const router = useRouter();
  const [token, setToken] = useState(initialToken);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<InviteValidateResponse | null>(null);
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!token) {
      setError("Missing invite token.");
      setLoading(false);
      return;
    }
    async function validate() {
      try {
        const res = await fetch(`/api/auth/invite/validate?token=${encodeURIComponent(token)}`);
        if (!res.ok) {
          const data = await res.json().catch(() => null);
          setError(data?.detail || "This invite link is invalid or has expired.");
        } else {
          const data = (await res.json()) as InviteValidateResponse;
          setInfo(data);
        }
      } catch (e) {
        setError("Could not validate invite link. Please try again later.");
      } finally {
        setLoading(false);
      }
    }
    validate();
  }, [token]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!token) return;
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }
    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const res = await fetch("/api/auth/invite/accept", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data?.detail || "Failed to set password.");
        return;
      }
      // On success, redirect to login page with success message
      const emailParam = info?.email ? `&email=${encodeURIComponent(info.email)}` : '';
      router.push(`/login?invite=success${emailParam}`);
    } catch (e) {
      setError("An unexpected error occurred.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Set your password</CardTitle>
          <CardDescription>
            {loading
              ? "Validating your invite..."
              : info
              ? `Account for ${info.email}`
              : "Invite status"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {loading ? (
            <p className="text-sm text-muted-foreground">Checking your invite link...</p>
          ) : info ? (
            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label className="block text-sm font-medium mb-1">New password</label>
                <Input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Confirm password</label>
                <Input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  required
                />
              </div>
              <Button type="submit" className="w-full" disabled={submitting}>
                {submitting ? "Setting password..." : "Set password and continue"}
              </Button>
            </form>
          ) : (
            <p className="text-sm text-muted-foreground">
              This invite link is invalid or has expired. Please contact your administrator.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}


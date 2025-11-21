'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    console.log('LOGIN PAGE HYDRATED - Build 2025-11-21-v3');
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    console.log('LOGIN HANDLE SUBMIT FIRED');
    setError('');
    setLoading(true);

    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });

      let data;
      try {
        data = await response.json();
      } catch (jsonError) {
        // If response is not JSON, use status text
        setError(response.statusText || 'Login failed');
        setLoading(false);
        return;
      }

      if (!response.ok) {
        // Handle error - ensure it's always a string
        let errorMessage = 'Invalid email or password';
        if (data?.error) {
          errorMessage = typeof data.error === 'string' ? data.error : String(data.error);
        } else if (data?.detail) {
          if (Array.isArray(data.detail)) {
            // Handle validation error arrays
            errorMessage = data.detail.map((err: any) => 
              `${err.loc?.join('.') || 'field'}: ${err.msg || 'Invalid value'}`
            ).join(', ');
          } else if (typeof data.detail === 'string') {
            errorMessage = data.detail;
          } else {
            errorMessage = String(data.detail);
          }
        }
        setError(errorMessage);
        setLoading(false);
        // Prevent form from submitting and causing page reload
        return;
      }

      const { user } = data;
      if (!user) {
        setError('Invalid response from server');
        setLoading(false);
        return;
      }

      // Cookie is already set by the API route (forwarded from backend)
      // No need to store anything in localStorage
      
      // Redirect based on user role
      // Admins go to /admin, regular users go to main chat
      // Use window.location for full page reload to ensure middleware sees the cookie
      const redirectPath = user.role === 'ADMIN' ? '/admin' : '/';
      window.location.href = redirectPath;
    } catch (err) {
      setError('An error occurred. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center p-4">
      <div className="mb-8 flex justify-center">
        <Image
          src="/asi-logo.png"
          alt="Arrow Systems Logo"
          width={200}
          height={80}
          className="h-auto w-auto object-contain"
          priority
        />
      </div>
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Sign in</CardTitle>
          <CardDescription>Enter your email and password to login</CardDescription>
          <p data-build-id="login-build-2025-11-21-v3" style={{fontSize: '10px', color: '#666'}}>
            Build: 2025-11-21-v3
          </p>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={loading}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                disabled={loading}
              />
            </div>
            {error && (
              <div className="rounded-md bg-destructive/10 p-3 text-sm text-destructive">
                {error}
              </div>
            )}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Signing in...' : 'Sign in'}
            </Button>
            <div className="text-center text-sm text-muted-foreground">
              Need an account? Contact your administrator.
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}


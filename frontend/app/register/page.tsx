'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Image from 'next/image';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

export default function RegisterPage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to login page after a brief delay
    const timer = setTimeout(() => {
      router.push('/login');
    }, 2000);
    return () => clearTimeout(timer);
  }, [router]);

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
          <CardTitle>Registration Disabled</CardTitle>
          <CardDescription>Public registration is not available</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Account creation is restricted to administrators only. Please contact your administrator to request an account.
          </p>
          <Button 
            onClick={() => router.push('/login')} 
            className="w-full"
          >
            Go to Login
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}

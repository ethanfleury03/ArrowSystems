'use client';

import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { useState } from 'react';

export default function LogoutButton() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);

  const handleLogout = async () => {
    setLoading(true);
    try {
      // Clear localStorage first
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_token');
        localStorage.removeItem('user_profile');
      }
      
      await fetch('/api/auth/logout', { method: 'POST' });
      
      // Force full page reload to clear all state
      window.location.href = '/login';
    } catch (error) {
      console.error('Logout error:', error);
      window.location.href = '/login';
    }
  };

  return (
    <Button onClick={handleLogout} variant="outline" className="w-full" disabled={loading}>
      {loading ? 'Logging out...' : 'Logout'}
    </Button>
  );
}


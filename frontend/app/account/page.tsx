import { redirect } from 'next/navigation';
import { extractJwtFromCookie } from '@/lib/authClient';
import { iamBackendGet } from '@/lib/iam-backend';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import LogoutButton from './logout-button';

async function getUserFromSession() {
  // Extract JWT from cookie
  const token = await extractJwtFromCookie();
  
  if (!token) {
    return null;
  }
  
  try {
    // Call backend with JWT in Authorization header
    const response = await iamBackendGet('/auth/me', {
      'Authorization': `Bearer ${token}`,
    });

    if (!response.ok) {
      return null;
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching user from session:', error);
    return null;
  }
}

export default async function AccountPage() {
  const user = await getUserFromSession();

  if (!user) {
    redirect('/login');
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Account</CardTitle>
          <CardDescription>Your account information</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="text-sm font-medium text-muted-foreground">Email</div>
            <div className="text-lg">{user.email}</div>
          </div>
          <div className="space-y-2">
            <div className="text-sm font-medium text-muted-foreground">Role</div>
            <div className="text-lg capitalize">{user.role.toLowerCase()}</div>
          </div>
          <div className="space-y-2">
            <div className="text-sm font-medium text-muted-foreground">Member since</div>
            <div className="text-lg">
              {new Date(user.createdAt).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </div>
          </div>
          <div className="pt-4">
            <LogoutButton />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}


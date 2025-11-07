import { redirect } from 'next/navigation';
import { getUserFromSession } from '@/lib/auth';
import { AdminPanel } from '@/components/admin/admin-panel';

export default async function AdminPage() {
  // Check authentication and admin role
  const user = await getUserFromSession();
  
  if (!user) {
    redirect('/login');
  }
  
  if (user.role !== 'ADMIN') {
    redirect('/');
  }
  
  return <AdminPanel />;
}


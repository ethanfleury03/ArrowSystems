import { InviteAcceptForm } from './invite-form';

interface AcceptInvitePageProps {
  searchParams: { token?: string };
}

export default function AcceptInvitePage({ searchParams }: AcceptInvitePageProps) {
  const token = searchParams.token ?? '';
  return <InviteAcceptForm initialToken={token} />;
}


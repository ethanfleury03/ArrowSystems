import { fetchConversationDetails } from "@/lib/api/queryInsights";
import { notFound } from "next/navigation";
import { ConversationView } from "./_components/conversation-view";

interface Props {
  params: {
    customerId: string;
    conversationId: string;
  };
}

export default async function ConversationPage({ params }: Props) {
  const { conversationId } = params;

  const convo = await fetchConversationDetails(conversationId).catch(
    () => null
  );
  if (!convo) return notFound();

  return <ConversationView conversation={convo} />;
}


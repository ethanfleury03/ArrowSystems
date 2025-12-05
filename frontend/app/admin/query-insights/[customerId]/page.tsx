import { fetchCustomerQueries } from "@/lib/api/queryInsights";
import { notFound } from "next/navigation";
import { CustomerQueryList } from "./_components/customer-query-list";

export const dynamic = 'force-dynamic';

interface Props {
  params: { customerId: string };
  searchParams: { search?: string };
}

export default async function CustomerQueriesPage({ params, searchParams }: Props) {
  const { customerId } = params;
  const search = searchParams.search;

  const data = await fetchCustomerQueries(customerId, search).catch(() => null);
  if (!data) return notFound();

  return <CustomerQueryList data={data} initialSearch={search ?? ""} />;
}


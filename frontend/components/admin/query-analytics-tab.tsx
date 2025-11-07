'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useToast } from '@/hooks/use-toast';
import { CheckCircle2, XCircle, Eye, Download, Filter } from 'lucide-react';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination';

interface Query {
  query_id: string;
  query_text: string;
  session_id: string;
  timestamp: string;
  answer_text: string;
  documents_retrieved: string[];
  document_count: number;
  relevance_score: number | null;
  confidence: number | null;
  response_time_ms: number | null;
  matched_machine_name: string | null;
  is_resolved: boolean;
  sources: Array<{
    name: string;
    pages: string;
    snippet?: string;
  }>;
}

export function QueryAnalyticsTab() {
  const [activeTab, setActiveTab] = useState<'all' | 'failed'>('all');
  const [queries, setQueries] = useState<Query[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const [selectedQuery, setSelectedQuery] = useState<Query | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  
  // Filters
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [machineType, setMachineType] = useState('');
  const [minConfidence, setMinConfidence] = useState('');
  const [maxConfidence, setMaxConfidence] = useState('');
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('desc');
  const [includeResolved, setIncludeResolved] = useState(false);
  
  const { toast } = useToast();

  const fetchQueries = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        limit: pageSize.toString(),
        offset: ((currentPage - 1) * pageSize).toString(),
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      
      if (startDate) params.append('start_date', startDate);
      if (endDate) params.append('end_date', endDate);
      if (machineType) params.append('machine_type', machineType);
      if (minConfidence) params.append('min_confidence', minConfidence);
      if (maxConfidence) params.append('max_confidence', maxConfidence);
      if (activeTab === 'failed' && includeResolved) params.append('include_resolved', 'true');
      
      const endpoint = activeTab === 'failed' 
        ? `/api/admin/queries/failed?${params.toString()}`
        : `/api/admin/queries?${params.toString()}`;
      
      const response = await fetch(endpoint);
      if (!response.ok) throw new Error('Failed to fetch queries');
      
      const data = await response.json();
      setQueries(data.queries || []);
      setTotal(data.total || 0);
    } catch (error) {
      console.error('Error fetching queries:', error);
      toast({
        title: 'Error',
        description: 'Failed to load queries',
      });
    } finally {
      setLoading(false);
    }
  }, [activeTab, currentPage, startDate, endDate, machineType, minConfidence, maxConfidence, sortBy, sortOrder, includeResolved, pageSize, toast]);

  useEffect(() => {
    fetchQueries();
  }, [fetchQueries]);

  const handleMarkResolved = async (queryId: string) => {
    try {
      const response = await fetch('/api/admin/queries/mark_resolved', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query_id: queryId }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to mark query as resolved');
      }

      toast({
        title: 'Success',
        description: 'Query marked as resolved',
      });

      fetchQueries();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to mark query as resolved',
      });
    }
  };

  const handleExport = async (format: 'csv' | 'json') => {
    try {
      const params = new URLSearchParams({
        limit: '10000',  // Get all for export
        offset: '0',
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      
      if (startDate) params.append('start_date', startDate);
      if (endDate) params.append('end_date', endDate);
      if (machineType) params.append('machine_type', machineType);
      
      const endpoint = activeTab === 'failed' 
        ? `/api/admin/queries/failed?${params.toString()}`
        : `/api/admin/queries?${params.toString()}`;
      
      const response = await fetch(endpoint);
      if (!response.ok) throw new Error('Failed to fetch queries for export');
      
      const data = await response.json();
      const exportQueries = data.queries || [];
      
      if (format === 'csv') {
        // Convert to CSV
        const headers = ['Query', 'Timestamp', 'Confidence', 'Documents', 'Machine', 'Response Time (ms)'];
        const rows = exportQueries.map((q: Query) => [
          q.query_text,
          q.timestamp,
          q.confidence?.toFixed(2) || 'N/A',
          q.document_count,
          q.matched_machine_name || 'N/A',
          q.response_time_ms || 'N/A',
        ]);
        
        const csv = [headers.join(','), ...rows.map((r: any[]) => r.map(c => `"${c}"`).join(','))].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `queries_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
      } else {
        // Export as JSON
        const json = JSON.stringify(exportQueries, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `queries_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
      }
      
      toast({
        title: 'Success',
        description: `Exported ${exportQueries.length} queries`,
      });
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to export queries',
      });
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-semibold">Query Analytics</h2>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleExport('csv')}
          >
            <Download className="mr-2 h-4 w-4" />
            Export CSV
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleExport('json')}
          >
            <Download className="mr-2 h-4 w-4" />
            Export JSON
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b">
        <Button
          variant={activeTab === 'all' ? 'default' : 'ghost'}
          onClick={() => {
            setActiveTab('all');
            setCurrentPage(1);
          }}
        >
          All Queries ({total})
        </Button>
        <Button
          variant={activeTab === 'failed' ? 'default' : 'ghost'}
          onClick={() => {
            setActiveTab('failed');
            setCurrentPage(1);
          }}
        >
          Failed Queries
        </Button>
      </div>

      {/* Filters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 border rounded-lg">
        <div>
          <Label htmlFor="start-date">Start Date</Label>
          <Input
            id="start-date"
            type="date"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="end-date">End Date</Label>
          <Input
            id="end-date"
            type="date"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="machine-type">Machine Type</Label>
          <Input
            id="machine-type"
            placeholder="Filter by machine"
            value={machineType}
            onChange={(e) => setMachineType(e.target.value)}
          />
        </div>
        <div>
          <Label htmlFor="sort-by">Sort By</Label>
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger id="sort-by">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="timestamp">Timestamp</SelectItem>
              <SelectItem value="confidence">Confidence</SelectItem>
              <SelectItem value="relevance_score">Relevance</SelectItem>
              <SelectItem value="document_count">Document Count</SelectItem>
              <SelectItem value="frequency">Frequency</SelectItem>
            </SelectContent>
          </Select>
        </div>
        {activeTab === 'failed' && (
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="include-resolved"
              checked={includeResolved}
              onChange={(e) => setIncludeResolved(e.target.checked)}
            />
            <Label htmlFor="include-resolved">Include Resolved</Label>
          </div>
        )}
      </div>

      {/* Table */}
      {loading ? (
        <div className="text-center py-8">Loading queries...</div>
      ) : queries.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          No queries found
        </div>
      ) : (
        <>
          <div className="border rounded-lg">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Query</TableHead>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Documents</TableHead>
                  <TableHead>Machine</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {queries.map((query) => (
                  <TableRow key={query.query_id}>
                    <TableCell className="max-w-md">
                      <div className="truncate" title={query.query_text}>
                        {query.query_text}
                      </div>
                    </TableCell>
                    <TableCell>
                      {new Date(query.timestamp).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      {query.confidence !== null ? (
                        <Badge variant={query.confidence < 0.5 ? 'destructive' : 'default'}>
                          {(query.confidence * 100).toFixed(0)}%
                        </Badge>
                      ) : (
                        'N/A'
                      )}
                    </TableCell>
                    <TableCell>{query.document_count}</TableCell>
                    <TableCell>{query.matched_machine_name || '-'}</TableCell>
                    <TableCell>
                      {query.is_resolved ? (
                        <Badge variant="secondary">Resolved</Badge>
                      ) : query.confidence !== null && query.confidence < 0.5 ? (
                        <Badge variant="destructive">Failed</Badge>
                      ) : (
                        <Badge variant="default">OK</Badge>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => {
                            setSelectedQuery(query);
                            setDetailDialogOpen(true);
                          }}
                          title="View Details"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        {activeTab === 'failed' && !query.is_resolved && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => handleMarkResolved(query.query_id)}
                            title="Mark as Resolved"
                          >
                            <CheckCircle2 className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <Pagination>
              <PaginationContent>
                <PaginationPrevious
                  onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                  className={currentPage === 1 ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                />
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const page = currentPage <= 3 ? i + 1 : currentPage - 2 + i;
                  if (page > totalPages) return null;
                  return (
                    <PaginationItem key={page}>
                      <PaginationLink
                        onClick={() => setCurrentPage(page)}
                        isActive={currentPage === page}
                      >
                        {page}
                      </PaginationLink>
                    </PaginationItem>
                  );
                })}
                <PaginationNext
                  onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
                  className={currentPage === totalPages ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                />
              </PaginationContent>
            </Pagination>
          )}
        </>
      )}

      {/* Detail Dialog */}
      <Dialog open={detailDialogOpen} onOpenChange={setDetailDialogOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Query Details</DialogTitle>
            <DialogDescription>
              Full query information and retrieved documents
            </DialogDescription>
          </DialogHeader>
          {selectedQuery && (
            <div className="space-y-4">
              <div>
                <Label className="font-semibold">Query:</Label>
                <p className="mt-1">{selectedQuery.query_text}</p>
              </div>
              <div>
                <Label className="font-semibold">Answer:</Label>
                <p className="mt-1">{selectedQuery.answer_text}</p>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="font-semibold">Confidence:</Label>
                  <p>{selectedQuery.confidence ? `${(selectedQuery.confidence * 100).toFixed(1)}%` : 'N/A'}</p>
                </div>
                <div>
                  <Label className="font-semibold">Response Time:</Label>
                  <p>{selectedQuery.response_time_ms ? `${selectedQuery.response_time_ms}ms` : 'N/A'}</p>
                </div>
                <div>
                  <Label className="font-semibold">Documents Retrieved:</Label>
                  <p>{selectedQuery.document_count}</p>
                </div>
                <div>
                  <Label className="font-semibold">Machine:</Label>
                  <p>{selectedQuery.matched_machine_name || 'N/A'}</p>
                </div>
              </div>
              <div>
                <Label className="font-semibold">Retrieved Documents:</Label>
                <ul className="mt-1 list-disc list-inside">
                  {selectedQuery.documents_retrieved.map((doc, idx) => (
                    <li key={idx}>{doc}</li>
                  ))}
                </ul>
              </div>
              {selectedQuery.sources && selectedQuery.sources.length > 0 && (
                <div>
                  <Label className="font-semibold">Sources:</Label>
                  <div className="mt-2 space-y-2">
                    {selectedQuery.sources.map((source, idx) => (
                      <div key={idx} className="border rounded p-2">
                        <p className="font-medium">{source.name}</p>
                        {source.pages && <p className="text-sm text-muted-foreground">Pages: {source.pages}</p>}
                        {source.snippet && (
                          <p className="text-sm mt-1">{source.snippet}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setDetailDialogOpen(false)}>
              Close
            </Button>
            {selectedQuery && !selectedQuery.is_resolved && activeTab === 'failed' && (
              <Button onClick={() => {
                handleMarkResolved(selectedQuery.query_id);
                setDetailDialogOpen(false);
              }}>
                Mark as Resolved
              </Button>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}


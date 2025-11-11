'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination';
import { Badge } from '@/components/ui/badge';
import { RefreshCw, Trash2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Chunk {
  chunk_id: string;
  doc_title: string;
  chunk_text: string;
  summary_exists: boolean;
  embedding_exists: boolean;
  page_label: string | null;
  content_type: string | null;
}

export function ChunkViewerTab() {
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [regenerating, setRegenerating] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchChunks = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(`/api/admin/chunks?page=${page}&page_size=50`);
      if (!response.ok) throw new Error('Failed to fetch chunks');
      const data = await response.json();
      setChunks(data.chunks || []);
      setTotalPages(data.total_pages || 1);
    } catch (error) {
      console.error('Error fetching chunks:', error);
      toast({
        title: 'Error',
        description: 'Failed to load chunks',
      });
    } finally {
      setLoading(false);
    }
  }, [page, toast]);

  useEffect(() => {
    fetchChunks();
  }, [fetchChunks]);

  const handleRegenerateSummary = async (chunkId: string) => {
    try {
      setRegenerating(chunkId);
      const response = await fetch(`/api/admin/chunks/${chunkId}/regenerate-summary`, {
        method: 'POST',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to regenerate summary');
      }
      
      toast({
        title: 'Success',
        description: 'Summary regenerated successfully',
      });
      
      fetchChunks();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to regenerate summary',
      });
    } finally {
      setRegenerating(null);
    }
  };

  const handleDelete = async (chunkId: string) => {
    try {
      const response = await fetch(`/api/admin/chunks/${chunkId}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete chunk');
      }
      
      toast({
        title: 'Info',
        description: 'Chunk deletion requires re-indexing. Please remove the source document and re-run python -m backend.ingest',
      });
      
      fetchChunks();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to delete chunk',
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-semibold">Chunk Viewer</h2>
        <Button onClick={fetchChunks} variant="outline" size="sm">
          Refresh
        </Button>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading chunks...</div>
      ) : chunks.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          No chunks found
        </div>
      ) : (
        <>
          <div className="border rounded-lg">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Chunk ID</TableHead>
                  <TableHead>Document</TableHead>
                  <TableHead>Text Preview</TableHead>
                  <TableHead>Summary</TableHead>
                  <TableHead>Embedding</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {chunks.map((chunk) => (
                  <TableRow key={chunk.chunk_id}>
                    <TableCell className="font-mono text-xs">
                      {chunk.chunk_id.slice(0, 8)}...
                    </TableCell>
                    <TableCell>
                      <div className="max-w-[200px] truncate">
                        {chunk.doc_title}
                      </div>
                      {chunk.page_label && (
                        <div className="text-xs text-muted-foreground">
                          Page: {chunk.page_label}
                        </div>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="max-w-[300px] truncate text-sm text-muted-foreground">
                        {chunk.chunk_text}
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={chunk.summary_exists ? 'default' : 'secondary'}>
                        {chunk.summary_exists ? 'Yes' : 'No'}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={chunk.embedding_exists ? 'default' : 'secondary'}>
                        {chunk.embedding_exists ? 'Yes' : 'No'}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleRegenerateSummary(chunk.chunk_id)}
                          disabled={regenerating === chunk.chunk_id}
                          title="Regenerate Summary"
                        >
                          <RefreshCw className={`h-4 w-4 ${regenerating === chunk.chunk_id ? 'animate-spin' : ''}`} />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => handleDelete(chunk.chunk_id)}
                          title="Delete Chunk"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>

          {totalPages > 1 && (
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={() => setPage(Math.max(1, page - 1))}
                    className={page === 1 ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                  />
                </PaginationItem>
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const pageNum = Math.max(1, Math.min(totalPages - 4, page - 2)) + i;
                  if (pageNum > totalPages) return null;
                  return (
                    <PaginationItem key={pageNum}>
                      <PaginationLink
                        onClick={() => setPage(pageNum)}
                        isActive={pageNum === page}
                        className="cursor-pointer"
                      >
                        {pageNum}
                      </PaginationLink>
                    </PaginationItem>
                  );
                })}
                <PaginationItem>
                  <PaginationNext
                    onClick={() => setPage(Math.min(totalPages, page + 1))}
                    className={page === totalPages ? 'pointer-events-none opacity-50' : 'cursor-pointer'}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          )}
        </>
      )}
    </div>
  );
}


'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Sparkles } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface MissingSummaryChunk {
  chunk_id: string;
  doc_title: string;
  chunk_text: string;
  page_label: string | null;
  content_type: string | null;
}

export function SummariesTab() {
  const [chunks, setChunks] = useState<MissingSummaryChunk[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    fetchMissingSummaries();
  }, []);

  const fetchMissingSummaries = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/admin/summaries/missing');
      if (!response.ok) throw new Error('Failed to fetch missing summaries');
      const data = await response.json();
      setChunks(data.chunks || []);
    } catch (error) {
      console.error('Error fetching missing summaries:', error);
      toast({
        title: 'Error',
        description: 'Failed to load missing summaries',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleBatchGenerate = async () => {
    try {
      setGenerating(true);
      const response = await fetch('/api/admin/summaries/generate-batch', {
        method: 'POST',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to generate summaries');
      }
      
      const data = await response.json();
      
      toast({
        title: 'Success',
        description: `Generated ${data.generated} summaries out of ${data.total} chunks`,
      });
      
      fetchMissingSummaries();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to generate summaries',
      });
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-semibold">Missing Summaries</h2>
          <p className="text-sm text-muted-foreground mt-1">
            {chunks.length} chunks without summaries
          </p>
        </div>
        <Button
          onClick={handleBatchGenerate}
          disabled={generating || chunks.length === 0}
        >
          <Sparkles className="mr-2 h-4 w-4" />
          {generating ? 'Generating...' : 'Generate All Summaries'}
        </Button>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading chunks...</div>
      ) : chunks.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <p className="text-lg font-medium mb-2">All chunks have summaries!</p>
          <p className="text-sm">No action needed.</p>
        </div>
      ) : (
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Chunk ID</TableHead>
                <TableHead>Document</TableHead>
                <TableHead>Text Preview</TableHead>
                <TableHead>Page</TableHead>
                <TableHead>Type</TableHead>
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
                  </TableCell>
                  <TableCell>
                    <div className="max-w-[400px] truncate text-sm text-muted-foreground">
                      {chunk.chunk_text}
                    </div>
                  </TableCell>
                  <TableCell>
                    {chunk.page_label || 'N/A'}
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary">
                      {chunk.content_type || 'text'}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
}


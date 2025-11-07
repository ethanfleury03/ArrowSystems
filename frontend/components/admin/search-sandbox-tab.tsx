'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Search, Loader2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface SearchResult {
  query: string;
  retrieved_chunks: Array<{
    doc_id: string;
    pages: string;
    content_type: string;
    source_id: string;
  }>;
  machine_detection_fired: boolean;
  matched_machine_name: string | null;
  document_ids: string[];
  total_chunks: number;
}

export function SearchSandboxTab() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult | null>(null);
  const { toast } = useToast();

  const handleSearch = async () => {
    if (!query.trim()) {
      toast({
        title: 'Error',
        description: 'Please enter a query',
      });
      return;
    }

    try {
      setLoading(true);
      const response = await fetch('/api/admin/search-sandbox', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: 10,
          alpha: 0.5,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Search failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Search failed',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold mb-2">Search Sandbox</h2>
        <p className="text-sm text-muted-foreground">
          Test search queries and inspect retrieval results for debugging
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Test Query</CardTitle>
          <CardDescription>
            Enter a query to see detailed retrieval information
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="flex-1">
              <Label htmlFor="query-input" className="sr-only">
                Search Query
              </Label>
              <Input
                id="query-input"
                placeholder="Enter your query..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !loading) {
                    handleSearch();
                  }
                }}
              />
            </div>
            <Button onClick={handleSearch} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Search
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {results && (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Search Results</CardTitle>
              <CardDescription>
                Query: "{results.query}" • {results.total_chunks} chunks retrieved
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label className="text-sm font-medium">Machine Detection</Label>
                  <div className="mt-1">
                    <Badge variant={results.machine_detection_fired ? 'default' : 'secondary'}>
                      {results.machine_detection_fired ? 'Fired' : 'Not Fired'}
                    </Badge>
                    {results.matched_machine_name && (
                      <p className="text-sm text-muted-foreground mt-1">
                        Matched: {results.matched_machine_name}
                      </p>
                    )}
                  </div>
                </div>
                <div>
                  <Label className="text-sm font-medium">Document IDs</Label>
                  <div className="mt-1">
                    <p className="text-sm">{results.document_ids.length} unique documents</p>
                  </div>
                </div>
              </div>

              <div>
                <Label className="text-sm font-medium mb-2 block">Retrieved Chunks</Label>
                <div className="border rounded-lg divide-y">
                  {results.retrieved_chunks.map((chunk, idx) => (
                    <div key={idx} className="p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <p className="font-medium">{chunk.doc_id}</p>
                          <p className="text-sm text-muted-foreground">
                            Pages: {chunk.pages} • Type: {chunk.content_type}
                          </p>
                        </div>
                        <Badge variant="outline">#{idx + 1}</Badge>
                      </div>
                      <p className="text-xs text-muted-foreground font-mono">
                        Source ID: {chunk.source_id}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <Label className="text-sm font-medium mb-2 block">Document IDs (JSON)</Label>
                <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
                  {JSON.stringify(results.document_ids, null, 2)}
                </pre>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}


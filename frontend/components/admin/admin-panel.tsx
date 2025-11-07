'use client';

import { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DocumentsTab } from './documents-tab';
import { ChunkViewerTab } from './chunk-viewer-tab';
import { SummariesTab } from './summaries-tab';
import { SearchSandboxTab } from './search-sandbox-tab';
import { QueryAnalyticsTab } from './query-analytics-tab';

export function AdminPanel() {
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Admin Control Panel</h1>
        <p className="text-muted-foreground mt-2">
          Manage documents, chunks, summaries, and test search functionality
        </p>
      </div>
      
      <Tabs defaultValue="documents" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="documents">Documents</TabsTrigger>
          <TabsTrigger value="chunks">Chunk Viewer</TabsTrigger>
          <TabsTrigger value="summaries">Summaries</TabsTrigger>
          <TabsTrigger value="sandbox">Search Sandbox</TabsTrigger>
          <TabsTrigger value="analytics">Query Analytics</TabsTrigger>
        </TabsList>
        
        <TabsContent value="documents" className="mt-6">
          <DocumentsTab />
        </TabsContent>
        
        <TabsContent value="chunks" className="mt-6">
          <ChunkViewerTab />
        </TabsContent>
        
        <TabsContent value="summaries" className="mt-6">
          <SummariesTab />
        </TabsContent>
        
        <TabsContent value="sandbox" className="mt-6">
          <SearchSandboxTab />
        </TabsContent>
        
        <TabsContent value="analytics" className="mt-6">
          <QueryAnalyticsTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}


'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Input } from '@/components/ui/input';
import { Upload, Trash2, FileText, Edit, Power, PowerOff, Eye } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

interface Document {
  filename: string;
  size_bytes: number;
  uploaded_date: string | null;
  chunk_count: number;
  page_count: number;
  file_path: string;
  file_type: string;
  is_active: boolean;
  machine_model: string | null;
  category: string | null;
  product_family: string | null;
}

interface DocumentChunk {
  chunk_id: string;
  chunk_text: string;
  page_label: string | null;
  content_type: string | null;
}

export function DocumentsTab() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [chunksSheetOpen, setChunksSheetOpen] = useState(false);
  const [documentChunks, setDocumentChunks] = useState<DocumentChunk[]>([]);
  const [uploading, setUploading] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingDoc, setEditingDoc] = useState<Document | null>(null);
  const { toast } = useToast();

  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/admin/documents');
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
      toast({
        title: 'Error',
        description: 'Failed to load documents',
      });
    } finally {
      setLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const handleDelete = async () => {
    if (!selectedDoc) return;
    
    try {
      const response = await fetch(`/api/admin/documents/${encodeURIComponent(selectedDoc.filename)}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete document');
      }
      
      toast({
        title: 'Success',
        description: `Document ${selectedDoc.filename} deleted`,
      });
      
      setDeleteDialogOpen(false);
      setSelectedDoc(null);
      fetchDocuments();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to delete document',
      });
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['.pdf', '.docx', '.md', '.markdown'];
    const fileExt = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(fileExt)) {
      toast({
        title: 'Invalid file type',
        description: `Allowed types: ${allowedTypes.join(', ')}`,
      });
      return;
    }

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', file);

      // Show initial upload toast
      const uploadToast = toast({
        title: 'Uploading...',
        description: `Uploading ${file.name}...`,
      });

      const response = await fetch('/api/admin/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to upload document');
      }

      const data = await response.json();
      
      // Show success toast with details
      toast({
        title: 'Success',
        description: `${data.filename} uploaded and ingested successfully. ${data.chunk_count} chunks created from ${data.page_count} pages.`,
      });

      // Refresh document list to show new document
      await fetchDocuments();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to upload document',
      });
    } finally {
      setUploading(false);
      // Reset input
      event.target.value = '';
    }
  };

  const handleToggleStatus = async (doc: Document) => {
    try {
      const response = await fetch(`/api/admin/documents/${encodeURIComponent(doc.filename)}/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: !doc.is_active }),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to toggle document status');
      }
      
      toast({
        title: 'Success',
        description: `Document ${doc.filename} ${!doc.is_active ? 'enabled' : 'disabled'}`,
      });
      
      await fetchDocuments();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to toggle document status',
      });
    }
  };

  const handleEditMetadata = async (updates: { machine_model?: string; category?: string; product_family?: string }) => {
    if (!editingDoc) return;
    
    try {
      const response = await fetch(`/api/admin/documents/${encodeURIComponent(editingDoc.filename)}/metadata`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to update metadata');
      }
      
      toast({
        title: 'Success',
        description: `Metadata updated for ${editingDoc.filename}`,
      });
      
      await fetchDocuments();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to update metadata',
      });
    }
  };

  const handleRowClick = async (doc: Document) => {
    setSelectedDoc(doc);
    try {
      const response = await fetch(`/api/admin/documents/${encodeURIComponent(doc.filename)}/chunks`);
      if (!response.ok) throw new Error('Failed to fetch chunks');
      const data = await response.json();
      setDocumentChunks(data.chunks || []);
      setChunksSheetOpen(true);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to load document chunks',
      });
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-semibold">Documents</h2>
        <div className="flex gap-2">
          <Input
            type="file"
            accept=".pdf,.docx,.md,.markdown"
            onChange={handleUpload}
            disabled={uploading}
            className="hidden"
            id="file-upload"
          />
          <Button
            onClick={() => document.getElementById('file-upload')?.click()}
            disabled={uploading}
          >
            <Upload className="mr-2 h-4 w-4" />
            {uploading ? 'Processing...' : 'Upload Document'}
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-8">Loading documents...</div>
      ) : documents.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          No documents found. Upload a document to get started.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {documents.map((doc) => (
            <Card key={doc.filename} className="flex flex-col">
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <FileText className="h-5 w-5 text-muted-foreground flex-shrink-0" />
                    <CardTitle className="text-lg truncate" title={doc.filename}>
                      {doc.filename}
                    </CardTitle>
                  </div>
                  <Badge variant={doc.is_active ? 'default' : 'secondary'} className="ml-2 flex-shrink-0">
                    {doc.is_active ? 'Enabled' : 'Disabled'}
                  </Badge>
                </div>
                {doc.machine_model && (
                  <CardDescription className="mt-1">
                    Machine: {doc.machine_model}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="flex-1">
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Type:</span>
                    <Badge variant="outline">{doc.file_type.toUpperCase()}</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Pages:</span>
                    <span className="font-medium">{doc.page_count}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Chunks:</span>
                    <span className="font-medium">{doc.chunk_count}</span>
                  </div>
                  {doc.uploaded_date && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Last Ingestion:</span>
                      <span className="font-medium text-xs">
                        {new Date(doc.uploaded_date).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
              <CardFooter className="flex gap-2 pt-4">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => {
                    // Open PDF in new tab
                    const encodedFilename = encodeURIComponent(doc.filename);
                    window.open(`/api/documents/${encodedFilename}`, '_blank');
                  }}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  View
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  className="flex-1"
                  onClick={() => {
                    setSelectedDoc(doc);
                    setDeleteDialogOpen(true);
                  }}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </Button>
              </CardFooter>
            </Card>
          ))}
        </div>
      )}

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Document</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete &quot;{selectedDoc?.filename}&quot;? This action cannot be undone.
              You will need to re-index after deletion.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} className="bg-destructive text-destructive-foreground">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <Sheet open={chunksSheetOpen} onOpenChange={setChunksSheetOpen}>
        <SheetContent className="w-full sm:max-w-2xl overflow-y-auto">
          <SheetHeader>
            <SheetTitle>Document Chunks</SheetTitle>
            <SheetDescription>
              {selectedDoc?.filename} - {documentChunks.length} chunks
            </SheetDescription>
          </SheetHeader>
          <div className="mt-6 space-y-4">
            {documentChunks.map((chunk, idx) => (
              <div key={chunk.chunk_id} className="border rounded-lg p-4">
                <div className="flex justify-between items-start mb-2">
                  <span className="text-sm font-medium">Chunk {idx + 1}</span>
                  {chunk.page_label && (
                    <span className="text-xs text-muted-foreground">
                      Page: {chunk.page_label}
                    </span>
                  )}
                </div>
                <p className="text-sm text-muted-foreground line-clamp-3">
                  {chunk.chunk_text}
                </p>
                {chunk.content_type && (
                  <span className="text-xs text-muted-foreground mt-2 inline-block">
                    Type: {chunk.content_type}
                  </span>
                )}
              </div>
            ))}
          </div>
          </SheetContent>
        </Sheet>

        <Dialog open={editDialogOpen} onOpenChange={setEditDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Edit Document Metadata</DialogTitle>
              <DialogDescription>
                Update metadata for {editingDoc?.filename}
              </DialogDescription>
            </DialogHeader>
            {editingDoc && (
              <div className="space-y-4 py-4">
                <div>
                  <Label htmlFor="machine_model">Machine Model</Label>
                  <Input
                    id="machine_model"
                    defaultValue={editingDoc.machine_model || ''}
                    placeholder="e.g., 2800 Series Mini Laser Pro"
                    onBlur={(e) => {
                      if (e.target.value !== editingDoc.machine_model) {
                        handleEditMetadata({ machine_model: e.target.value || undefined });
                      }
                    }}
                  />
                </div>
                <div>
                  <Label htmlFor="category">Category</Label>
                  <Input
                    id="category"
                    defaultValue={editingDoc.category || ''}
                    placeholder="e.g., User Manual, Installation Guide"
                    onBlur={(e) => {
                      if (e.target.value !== editingDoc.category) {
                        handleEditMetadata({ category: e.target.value || undefined });
                      }
                    }}
                  />
                </div>
                <div>
                  <Label htmlFor="product_family">Product Family</Label>
                  <Input
                    id="product_family"
                    defaultValue={editingDoc.product_family || ''}
                    placeholder="e.g., DuraFlex, anyCUT"
                    onBlur={(e) => {
                      if (e.target.value !== editingDoc.product_family) {
                        handleEditMetadata({ product_family: e.target.value || undefined });
                      }
                    }}
                  />
                </div>
              </div>
            )}
            <DialogFooter>
              <Button variant="outline" onClick={() => setEditDialogOpen(false)}>
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    );
  }


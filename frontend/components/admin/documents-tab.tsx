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
import { Upload, Trash2, FileText, Edit, Power, PowerOff } from 'lucide-react';
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

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
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
  };

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
          No documents found
        </div>
      ) : (
        <div className="border rounded-lg">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Filename</TableHead>
                <TableHead>Machine Model</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Pages</TableHead>
                <TableHead>Chunks</TableHead>
                <TableHead>Last Ingestion</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {documents.map((doc) => (
                <TableRow
                  key={doc.filename}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => handleRowClick(doc)}
                >
                  <TableCell className="font-medium">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      {doc.filename}
                    </div>
                  </TableCell>
                  <TableCell>
                    {doc.machine_model || '-'}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{doc.file_type.toUpperCase()}</Badge>
                  </TableCell>
                  <TableCell>{doc.page_count}</TableCell>
                  <TableCell>{doc.chunk_count}</TableCell>
                  <TableCell>
                    {doc.uploaded_date ? new Date(doc.uploaded_date).toLocaleDateString() : 'N/A'}
                  </TableCell>
                  <TableCell>
                    <Badge variant={doc.is_active ? 'default' : 'secondary'}>
                      {doc.is_active ? 'Enabled' : 'Disabled'}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <div className="flex justify-end gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleToggleStatus(doc);
                        }}
                        title={doc.is_active ? 'Disable' : 'Enable'}
                      >
                        {doc.is_active ? (
                          <PowerOff className="h-4 w-4" />
                        ) : (
                          <Power className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingDoc(doc);
                          setEditDialogOpen(true);
                        }}
                        title="Edit Metadata"
                      >
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedDoc(doc);
                          setDeleteDialogOpen(true);
                        }}
                        title="Delete"
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
      )}

      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Document</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{selectedDoc?.filename}"? This action cannot be undone.
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


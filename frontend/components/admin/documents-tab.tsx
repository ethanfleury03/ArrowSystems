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
import { Upload, Trash2, FileText, Edit, Power, PowerOff, Eye, Plus } from 'lucide-react';
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';

interface Document {
  filename: string;
  size_bytes: number | null;
  uploaded_date: string | null;
  chunk_count: number;
  page_count: number;
  file_path: string | null;
  file_type: string | null;
  is_active: boolean;
  machine_model: string | null;
  category: string | null;
  product_family: string | null;
  ingestion_status?: string | null;
  ingestion_metadata_id?: string | null;
  ingestion_error?: string | null;
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
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedMachine, setSelectedMachine] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [machines, setMachines] = useState<string[]>([]);
  const [loadingMachines, setLoadingMachines] = useState(false);
  const [addMachineDialogOpen, setAddMachineDialogOpen] = useState(false);
  const [newMachineName, setNewMachineName] = useState('');
  const [addingMachine, setAddingMachine] = useState(false);
  const [pollingInterval, setPollingInterval] = useState<NodeJS.Timeout | null>(null);
  const [testMode, setTestMode] = useState(false);
  const { toast } = useToast();

  // Helper function to get status label
  const getStatusLabel = (status: string | null | undefined): string => {
    if (!status) return '';
    switch (status) {
      case 'PENDING_INGESTION':
        return 'Pending';
      case 'CHUNKING':
        return 'Processing (chunking)';
      case 'READY_FOR_EMBEDDING':
        return 'Ready for embeddings';
      case 'EMBEDDING':
        return 'Processing (embedding)';
      case 'COMPLETE':
        return 'Complete';
      case 'DELETING':
        return 'Deleting…';
      case 'REBUILDING_INDEX':
        return 'Rebuilding index…';
      case 'FAILED':
        return 'Failed';
      default:
        return status;
    }
  };

  // Helper function to get status badge variant
  const getStatusVariant = (status: string | null | undefined): 'default' | 'secondary' | 'destructive' | 'outline' => {
    if (!status) return 'outline';
    switch (status) {
      case 'PENDING_INGESTION':
        return 'secondary';
      case 'CHUNKING':
        return 'default';
      case 'READY_FOR_EMBEDDING':
        return 'default';
      case 'EMBEDDING':
        return 'default';
      case 'COMPLETE':
        return 'default';
      case 'DELETING':
        return 'default';
      case 'REBUILDING_INDEX':
        return 'default';
      case 'FAILED':
        return 'destructive';
      default:
        return 'outline';
    }
  };

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
    fetchMachines();
    checkTestMode();
  }, [fetchDocuments]);

  const checkTestMode = useCallback(async () => {
    try {
      const response = await fetch('/api/admin/test-mode');
      if (response.ok) {
        const data = await response.json();
        setTestMode(data.test_mode || false);
      }
    } catch (error) {
      console.error('Error checking test mode:', error);
    }
  }, []);

  // Poll for status updates on documents that are pending, chunking, embedding, or deleting
  useEffect(() => {
    const hasActiveIngestion = documents.some(
      doc => doc.ingestion_status === 'PENDING_INGESTION' || 
             doc.ingestion_status === 'CHUNKING' || 
             doc.ingestion_status === 'READY_FOR_EMBEDDING' ||
             doc.ingestion_status === 'EMBEDDING' ||
             doc.ingestion_status === 'DELETING' ||
             doc.ingestion_status === 'REBUILDING_INDEX'
    );

    if (hasActiveIngestion) {
      // Poll every 3 seconds
      const interval = setInterval(() => {
        fetchDocuments();
      }, 3000);

      return () => {
        clearInterval(interval);
      };
    }
  }, [documents, fetchDocuments]);

  const fetchMachines = useCallback(async () => {
    try {
      setLoadingMachines(true);
      const response = await fetch('/api/admin/machines');
      if (!response.ok) throw new Error('Failed to fetch machines');
      const data = await response.json();
      // Backend now returns array of objects with {id, name, document_count, created_at}
      // Extract just the names for the dropdown
      if (Array.isArray(data)) {
        setMachines(data.map((m: { name: string }) => m.name));
      } else if (data.machines && Array.isArray(data.machines)) {
        // Fallback for old format
        setMachines(data.machines);
      } else {
        setMachines([]);
      }
    } catch (error) {
      console.error('Error fetching machines:', error);
      toast({
        title: 'Error',
        description: 'Failed to load machine models',
      });
    } finally {
      setLoadingMachines(false);
    }
  }, [toast]);

  const handleAddMachine = async () => {
    if (!newMachineName.trim()) {
      toast({
        title: 'Error',
        description: 'Machine name cannot be empty',
      });
      return;
    }

    try {
      setAddingMachine(true);
      const response = await fetch('/api/admin/machines', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newMachineName.trim() }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create machine');
      }

      const data = await response.json();
      toast({
        title: 'Success',
        description: `Machine model "${data.name}" added successfully`,
      });

      // Refresh machines list
      await fetchMachines();
      
      // Auto-select the newly created machine
      setSelectedMachine(data.name);
      
      // Close dialog
      setAddMachineDialogOpen(false);
      setNewMachineName('');
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to add machine model',
      });
    } finally {
      setAddingMachine(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
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

    setSelectedFile(file);
    setUploadDialogOpen(true);
  };

  const handleDelete = async () => {
    if (!selectedDoc) return;

    // Check if we have metadata_id for Phase 4 deletion
    if (!selectedDoc.ingestion_metadata_id) {
      toast({
        title: 'Error',
        description: 'Document metadata ID not found. Cannot delete.',
        variant: 'destructive',
      });
      return;
    }

    try {
      const response = await fetch(`/api/admin/documents/metadata/${selectedDoc.ingestion_metadata_id}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to delete document');
      }
      
      toast({
        title: 'Deletion started',
        description: 'Document deletion started. The index is rebuilding in the background.',
      });
      
      setDeleteDialogOpen(false);
      setSelectedDoc(null);
      // Refresh to show DELETING status
      fetchDocuments();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to delete document',
        variant: 'destructive',
      });
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    if (!selectedMachine) {
      toast({
        title: 'Error',
        description: 'Please select a machine model',
      });
      return;
    }

    try {
      setUploading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('machine_model', selectedMachine);
      if (description.trim()) {
        formData.append('description', description.trim());
      }

      // Show initial upload toast
      toast({
        title: 'Uploading...',
        description: `Uploading ${selectedFile.name}...`,
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
        description: `${data.metadata.filename} uploaded successfully. Status: ${data.metadata.status}`,
      });

      // Reset form
      setSelectedFile(null);
      setSelectedMachine('');
      setDescription('');
      setUploadDialogOpen(false);

      // Refresh document list to show new document
      await fetchDocuments();
    } catch (error: any) {
      toast({
        title: 'Error',
        description: error.message || 'Failed to upload document',
      });
    } finally {
      setUploading(false);
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

  // Check if test mode is enabled (via environment variable or API)
  const [isTestMode, setIsTestMode] = useState(false);
  
  useEffect(() => {
    // Check test mode status
    const checkTestMode = async () => {
      try {
        // Test mode is determined by backend, we can check via a simple API call
        // or read from environment. For now, we'll check if test directories exist
        // by trying to fetch a test mode indicator from the backend
        const response = await fetch('/api/admin/test/mode-status');
        if (response.ok) {
          const data = await response.json();
          setIsTestMode(data.test_mode || false);
        }
      } catch (error) {
        // If endpoint doesn't exist yet, assume false
        setIsTestMode(false);
      }
    };
    checkTestMode();
  }, []);
  
  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <h2 className="text-2xl font-semibold">Documents</h2>
          {testMode && (
            <Badge variant="destructive" className="text-xs">
              TEST MODE (temporary index)
            </Badge>
          )}
        </div>
        <div className="flex gap-2">
          <Input
            type="file"
            accept=".pdf,.docx,.md,.markdown"
            onChange={handleFileSelect}
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
                {doc.ingestion_status && (
                  <div className="mt-1">
                    <Badge variant={getStatusVariant(doc.ingestion_status)} className="text-xs">
                      {getStatusLabel(doc.ingestion_status)}
                    </Badge>
                    {doc.ingestion_error && (
                      <p className="text-xs text-destructive mt-1" title={doc.ingestion_error}>
                        Error: {doc.ingestion_error.substring(0, 50)}...
                      </p>
                    )}
                  </div>
                )}
                {doc.machine_model && (
                  <CardDescription className="mt-1">
                    Machine: {doc.machine_model}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="flex-1">
                <div className="space-y-2 text-sm">
                  {doc.file_type && (
                    <div className="flex items-center justify-between">
                      <span className="text-muted-foreground">Type:</span>
                      <Badge variant="outline">{doc.file_type.toUpperCase()}</Badge>
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Pages:</span>
                    <span className="font-medium">{doc.page_count}</span>
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
              Are you sure you want to delete &quot;{selectedDoc?.filename}&quot;? This requires a full index rebuild.
              The deletion will happen in the background.
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

        {/* Upload Dialog */}
        <Dialog open={uploadDialogOpen} onOpenChange={setUploadDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Upload Document</DialogTitle>
              <DialogDescription>
                {selectedFile?.name}
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="machine-select">Machine Model *</Label>
                {loadingMachines ? (
                  <div className="text-sm text-muted-foreground mt-2">Loading machines...</div>
                ) : machines.length === 0 ? (
                  <div className="text-sm text-muted-foreground mt-2">
                    No machines available. Click the + button to add one.
                  </div>
                ) : null}
                <div className="flex gap-2 mt-2">
                  <Select value={selectedMachine} onValueChange={setSelectedMachine} disabled={loadingMachines}>
                    <SelectTrigger id="machine-select" className="flex-1">
                      <SelectValue placeholder={machines.length === 0 ? "No machines available" : "Select a machine model"} />
                    </SelectTrigger>
                    <SelectContent>
                      {machines.map((machine) => (
                        <SelectItem key={machine} value={machine}>
                          {machine}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setAddMachineDialogOpen(true)}
                    title="Add new machine model"
                    disabled={loadingMachines}
                  >
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div>
                <Label htmlFor="description">Description (Optional)</Label>
                <Textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Optional description for this document"
                  rows={3}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => {
                setUploadDialogOpen(false);
                setSelectedFile(null);
                setSelectedMachine('');
                setDescription('');
              }}>
                Cancel
              </Button>
              <Button onClick={handleUpload} disabled={uploading || !selectedMachine}>
                {uploading ? 'Uploading...' : 'Upload'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Add Machine Dialog */}
        <Dialog open={addMachineDialogOpen} onOpenChange={setAddMachineDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add New Machine Model</DialogTitle>
              <DialogDescription>
                Add a new machine model to the system
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div>
                <Label htmlFor="new-machine-name">Machine Name *</Label>
                <Input
                  id="new-machine-name"
                  value={newMachineName}
                  onChange={(e) => setNewMachineName(e.target.value)}
                  placeholder="e.g., NEW_MACHINE_MODEL"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && newMachineName.trim()) {
                      handleAddMachine();
                    }
                  }}
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => {
                setAddMachineDialogOpen(false);
                setNewMachineName('');
              }}>
                Cancel
              </Button>
              <Button onClick={handleAddMachine} disabled={addingMachine || !newMachineName.trim()}>
                {addingMachine ? 'Adding...' : 'Add Machine'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    );
  }


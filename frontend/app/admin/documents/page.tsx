"use client";

import { FileText } from "lucide-react";

export default function AdminDocumentsPage() {
  return (
    <div className="rounded-xl border border-dashed border-primary/30 bg-background p-10 text-center shadow-sm">
      <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
        <FileText className="h-8 w-8" />
      </div>
      <h1 className="mt-6 text-2xl font-semibold">Documents</h1>
      <p className="mt-2 text-sm text-muted-foreground">
        Documents coming soon. This section will provide document ingestion, status monitoring, and metadata controls.
      </p>
    </div>
  );
}

-- ArrowSystems Postgres schema upgrade: document_machine_models join table
-- Date: 2025-12-18
--
-- This migration is:
-- - Postgres-only
-- - Idempotent (safe to re-run)
-- - Best-effort backfill from legacy documents.machine_model and document_ingestion_metadata.machine_model
--
-- SAFETY:
-- - Does NOT touch GCS
-- - Does NOT drop/rename existing objects
-- - Uses ON CONFLICT DO NOTHING for backfill

BEGIN;

-- Fail fast if prerequisite tables are missing
DO $$
BEGIN
  IF to_regclass('public.documents') IS NULL THEN
    RAISE EXCEPTION 'Missing required table: public.documents';
  END IF;
  IF to_regclass('public.machine_models') IS NULL THEN
    RAISE EXCEPTION 'Missing required table: public.machine_models';
  END IF;
END $$;

-- A) Create join table
CREATE TABLE IF NOT EXISTS public.document_machine_models (
  document_id INTEGER NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
  machine_model_id INTEGER NOT NULL REFERENCES public.machine_models(id) ON DELETE CASCADE
);

-- Ensure unique constraint exists (table may pre-exist without it)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'uq_document_machine_models'
      AND conrelid = 'public.document_machine_models'::regclass
  ) THEN
    ALTER TABLE public.document_machine_models
      ADD CONSTRAINT uq_document_machine_models UNIQUE (document_id, machine_model_id);
  END IF;
END $$;

-- Helpful indexes (idempotent)
CREATE INDEX IF NOT EXISTS ix_document_machine_models_document_id
  ON public.document_machine_models (document_id);

CREATE INDEX IF NOT EXISTS ix_document_machine_models_machine_model_id
  ON public.document_machine_models (machine_model_id);

-- Helper: safe JSONB parse (returns NULL if invalid)
CREATE OR REPLACE FUNCTION public._try_parse_jsonb(p_text text)
RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
  RETURN p_text::jsonb;
EXCEPTION WHEN others THEN
  RETURN NULL;
END;
$$;

-- B) Backfill (best effort)

-- B1) documents.machine_model is a single name (non-JSON array)
INSERT INTO public.document_machine_models (document_id, machine_model_id)
SELECT d.id, mm.id
FROM public.documents d
JOIN public.machine_models mm ON trim(mm.name) = trim(d.machine_model)
WHERE d.machine_model IS NOT NULL
  AND trim(d.machine_model) <> ''
  AND left(trim(d.machine_model), 1) <> '['
ON CONFLICT (document_id, machine_model_id) DO NOTHING;

-- B2) documents.machine_model is a JSON array string like ["DuraFlex","DuraCore"]
INSERT INTO public.document_machine_models (document_id, machine_model_id)
SELECT d.id, mm.id
FROM public.documents d
JOIN LATERAL (SELECT public._try_parse_jsonb(d.machine_model) AS j) j ON true
JOIN LATERAL jsonb_array_elements_text(j.j) AS x(name) ON true
JOIN public.machine_models mm ON trim(mm.name) = trim(x.name)
WHERE d.machine_model IS NOT NULL
  AND left(trim(d.machine_model), 1) = '['
  AND j.j IS NOT NULL
  AND jsonb_typeof(j.j) = 'array'
ON CONFLICT (document_id, machine_model_id) DO NOTHING;

-- B3) Fallback backfill from document_ingestion_metadata.machine_model (guarded)
DO $$
BEGIN
  IF to_regclass('public.document_ingestion_metadata') IS NOT NULL
     AND EXISTS (
       SELECT 1 FROM information_schema.columns
       WHERE table_schema = 'public' AND table_name = 'document_ingestion_metadata' AND column_name = 'machine_model'
     )
     AND EXISTS (
       SELECT 1 FROM information_schema.columns
       WHERE table_schema = 'public' AND table_name = 'document_ingestion_metadata' AND column_name = 'filename'
     )
  THEN
    -- Prefer filling only when documents.machine_model is empty
    INSERT INTO public.document_machine_models (document_id, machine_model_id)
    SELECT d.id, mm.id
    FROM public.documents d
    JOIN public.document_ingestion_metadata dim ON dim.filename = d.file_name
    JOIN public.machine_models mm ON trim(mm.name) = trim(dim.machine_model)
    WHERE (d.machine_model IS NULL OR trim(d.machine_model) = '')
      AND dim.machine_model IS NOT NULL
      AND trim(dim.machine_model) <> ''
      AND left(trim(dim.machine_model), 1) <> '['
    ON CONFLICT (document_id, machine_model_id) DO NOTHING;

    INSERT INTO public.document_machine_models (document_id, machine_model_id)
    SELECT d.id, mm.id
    FROM public.documents d
    JOIN public.document_ingestion_metadata dim ON dim.filename = d.file_name
    JOIN LATERAL (SELECT public._try_parse_jsonb(dim.machine_model) AS j) j ON true
    JOIN LATERAL jsonb_array_elements_text(j.j) AS x(name) ON true
    JOIN public.machine_models mm ON trim(mm.name) = trim(x.name)
    WHERE (d.machine_model IS NULL OR trim(d.machine_model) = '')
      AND dim.machine_model IS NOT NULL
      AND left(trim(dim.machine_model), 1) = '['
      AND j.j IS NOT NULL
      AND jsonb_typeof(j.j) = 'array'
    ON CONFLICT (document_id, machine_model_id) DO NOTHING;
  END IF;
END $$;

COMMIT;

-- C) Verification queries (run automatically by psql, safe to re-run)
-- Table exists?
SELECT to_regclass('public.document_machine_models') AS document_machine_models_table;

-- Constraint exists?
SELECT conname, pg_get_constraintdef(oid) AS constraint_def
FROM pg_constraint
WHERE conrelid = 'public.document_machine_models'::regclass
  AND conname = 'uq_document_machine_models';

-- Row counts
SELECT count(*) AS document_machine_models_rows
FROM public.document_machine_models;

SELECT count(DISTINCT document_id) AS distinct_documents_mapped
FROM public.document_machine_models;

SELECT count(DISTINCT machine_model_id) AS distinct_machine_models_mapped
FROM public.document_machine_models;

-- Sample join
SELECT
  d.id AS document_id,
  d.file_name,
  array_agg(mm.name ORDER BY mm.name) AS machine_models
FROM public.documents d
JOIN public.document_machine_models dmm ON dmm.document_id = d.id
JOIN public.machine_models mm ON mm.id = dmm.machine_model_id
GROUP BY d.id, d.file_name
ORDER BY d.id DESC
LIMIT 10;



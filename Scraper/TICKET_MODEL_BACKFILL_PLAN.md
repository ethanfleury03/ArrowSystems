# Ticket Machine Model Backfill - Implementation Plan

## 1. Ticket Text Source
- **Primary**: `tickets_detail.conversation_json` 
  - Structure: `{request: {...}, messages: [{text, role, created_at, ...}]}`
  - Extract: `request.description` + all `messages[].text` (sorted by `created_at`)
- **Secondary**: `tickets_index.subject` (if available)
- **Build conversation blob**: subject + description + all message texts (chronological)

## 2. Machine Model Source
- **Database**: `machine_models` table (id, name, machine_kind)
- **No aliases field**: Generate aliases heuristically:
  - Case variants: "DuraFlex" → ["duraflex", "DURAFLEX"]
  - Spacing variants: "DuraFlex" → ["Dura Flex", "dura flex"]
  - Hyphen variants: "EZCut" → ["EZ-Cut", "ez-cut"]
  - Numeric extraction: "2800" → ["2800"]
- **Support**: Cloud SQL (via DATABASE_URL) OR JSON file input

## 3. Matching Logic
- **Normalization**: lowercase, whitespace collapse, punctuation removal
- **Word-boundary matching**: Use regex word boundaries
- **Scoring**:
  - Exact full name match = 100 points
  - Alias match = 80 points  
  - Partial token match = 50 points (only if unique)
- **False positive prevention**:
  - Minimum token length: 3 characters
  - Exclude common English words
  - Require word boundaries

## 4. Database Schema
- **`ticket_machine_model_matches`**: One row per match
  - ticket_id, machine_model_id, machine_model_name, match_source, score, evidence_snippet, created_at
- **`ticket_machine_model_assignment`**: One row per ticket (summary)
  - ticket_id, machine_model_ids (JSON), status (unassigned|assigned|ambiguous), confidence, method, updated_at

## 5. Implementation Files
- `utils/machine_models_loader.py` - Load models from DB/JSON
- `utils/machine_model_matcher.py` - Matching + scoring logic
- `scripts/backfill_ticket_machine_models.py` - Main CLI script
- `TICKET_MODEL_BACKFILL.md` - Usage documentation

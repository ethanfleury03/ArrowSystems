#!/bin/bash
# Database restore script
# Supports both SQLite and PostgreSQL

if [ -z "$1" ]; then
    echo "❌ Error: Backup file path required"
    echo "Usage: ./restore-db.sh <backup-file>"
    echo "Example: ./restore-db.sh backups/sqlite_backup_20240101_120000.db.gz"
    exit 1
fi

BACKUP_FILE="$1"
DATABASE_URL="${DATABASE_URL:-file:./dev.db}"

echo "🔄 Starting database restore..."
echo "📅 Date: $(date)"
echo "📁 Backup file: $BACKUP_FILE"
echo "📍 Database URL: $DATABASE_URL"

# Check if backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Error: Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Check if DATABASE_URL is SQLite or PostgreSQL
if [[ $DATABASE_URL == file:* ]]; then
    # SQLite restore
    DB_FILE="${DATABASE_URL#file:}"
    
    # Check if backup is compressed
    if [[ $BACKUP_FILE == *.gz ]]; then
        echo "📦 Decompressing backup..."
        gunzip -c "$BACKUP_FILE" > "${BACKUP_FILE%.gz}"
        TEMP_FILE="${BACKUP_FILE%.gz}"
    else
        TEMP_FILE="$BACKUP_FILE"
    fi
    
    # Create backup of current database before restore
    if [ -f "$DB_FILE" ]; then
        CURRENT_BACKUP="${DB_FILE}.pre-restore-$(date +%Y%m%d_%H%M%S)"
        cp "$DB_FILE" "$CURRENT_BACKUP"
        echo "💾 Current database backed up to: $CURRENT_BACKUP"
    fi
    
    # Restore
    cp "$TEMP_FILE" "$DB_FILE"
    echo "✅ SQLite database restored successfully"
    
    # Clean up temporary file if it was decompressed
    if [[ $BACKUP_FILE == *.gz ]]; then
        rm "$TEMP_FILE"
    fi
    
elif [[ $DATABASE_URL == postgresql* ]]; then
    # PostgreSQL restore
    echo "⚠️  WARNING: This will overwrite your current PostgreSQL database!"
    read -p "Are you sure you want to continue? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" != "yes" ]; then
        echo "❌ Restore cancelled"
        exit 0
    fi
    
    # Check if backup is compressed
    if [[ $BACKUP_FILE == *.gz ]]; then
        echo "📦 Decompressing backup..."
        TEMP_FILE="${BACKUP_FILE%.gz}"
        gunzip -c "$BACKUP_FILE" > "$TEMP_FILE"
    else
        TEMP_FILE="$BACKUP_FILE"
    fi
    
    # Restore using psql
    if command -v psql &> /dev/null; then
        psql "$DATABASE_URL" < "$TEMP_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✅ PostgreSQL database restored successfully"
        else
            echo "❌ Error: PostgreSQL restore failed"
            exit 1
        fi
    else
        echo "❌ Error: psql not found. Install PostgreSQL client tools."
        exit 1
    fi
    
    # Clean up temporary file if it was decompressed
    if [[ $BACKUP_FILE == *.gz ]]; then
        rm "$TEMP_FILE"
    fi
else
    echo "❌ Error: Unsupported database type in DATABASE_URL"
    exit 1
fi

echo "✅ Restore completed successfully!"


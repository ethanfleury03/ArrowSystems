#!/bin/bash
# Database backup script
# Supports both SQLite and PostgreSQL

BACKUP_DIR="./backups"
DATE=$(date +%Y%m%d_%H%M%S)
DATABASE_URL="${DATABASE_URL:-file:./dev.db}"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

echo "🔄 Starting database backup..."
echo "📅 Date: $(date)"
echo "📍 Database URL: $DATABASE_URL"

# Check if DATABASE_URL is SQLite or PostgreSQL
if [[ $DATABASE_URL == file:* ]]; then
    # SQLite backup
    DB_FILE="${DATABASE_URL#file:}"
    if [ -f "$DB_FILE" ]; then
        BACKUP_FILE="$BACKUP_DIR/sqlite_backup_$DATE.db"
        cp "$DB_FILE" "$BACKUP_FILE"
        echo "✅ SQLite backup created: $BACKUP_FILE"
        
        # Also create a compressed backup
        gzip -c "$BACKUP_FILE" > "$BACKUP_FILE.gz"
        echo "✅ Compressed backup created: $BACKUP_FILE.gz"
        
        # Remove uncompressed backup to save space
        rm "$BACKUP_FILE"
    else
        echo "❌ Error: SQLite database file not found: $DB_FILE"
        exit 1
    fi
elif [[ $DATABASE_URL == postgresql* ]]; then
    # PostgreSQL backup
    BACKUP_FILE="$BACKUP_DIR/postgres_backup_$DATE.sql"
    
    # Extract connection details from DATABASE_URL
    # Format: postgresql://user:password@host:port/database
    if command -v pg_dump &> /dev/null; then
        pg_dump "$DATABASE_URL" > "$BACKUP_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✅ PostgreSQL backup created: $BACKUP_FILE"
            
            # Compress the backup
            gzip "$BACKUP_FILE"
            echo "✅ Compressed backup created: $BACKUP_FILE.gz"
        else
            echo "❌ Error: PostgreSQL backup failed"
            exit 1
        fi
    else
        echo "❌ Error: pg_dump not found. Install PostgreSQL client tools."
        exit 1
    fi
else
    echo "❌ Error: Unsupported database type in DATABASE_URL"
    exit 1
fi

# Clean up old backups (keep last 30 days)
echo "🧹 Cleaning up old backups (keeping last 30 days)..."
find $BACKUP_DIR -name "*.db.gz" -mtime +30 -delete
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

echo "✅ Backup completed successfully!"
echo "📁 Backup location: $BACKUP_DIR"


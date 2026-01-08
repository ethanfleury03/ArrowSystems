#!/bin/bash
# Direct connection to Cloud SQL - NO PROXY NEEDED!
# This is simpler if Cloud SQL has external IP enabled

PASSWORD="${1:-C9ruwaOM4urx1dXLI8LxD5UpXoy_ii0cjRG7xlfdZX8}"
CLOUD_SQL_IP="${2:-}"

if [ -z "$CLOUD_SQL_IP" ]; then
    echo "⚠️  Need Cloud SQL IP address"
    echo ""
    echo "Find it in Google Cloud Console:"
    echo "  SQL → arrow-rag-support-prod → Overview → Connect → Public IP"
    echo ""
    echo "Or run this to try common patterns:"
    echo "  bash connect_direct.sh YOUR_PASSWORD <IP_ADDRESS>"
    exit 1
fi

export DATABASE_URL="postgresql://rag_user:${PASSWORD}@${CLOUD_SQL_IP}:5432/rag_app"

echo "🔧 Testing direct connection to ${CLOUD_SQL_IP}..."
python -c "
import sys
sys.path.insert(0, '.')
from backend.utils.db import get_engine
try:
    with get_engine().connect() as conn:
        conn.execute('SELECT 1')
    print('✅ Database connected directly! No proxy needed!')
except Exception as e:
    print(f'❌ Connection failed: {e}')
    print('')
    print('Possible issues:')
    print('  1. Cloud SQL instance needs external IP enabled')
    print('  2. Your RunPod IP needs to be whitelisted in Cloud SQL authorized networks')
    print('  3. Wrong IP address')
    sys.exit(1)
" && echo "" && echo "🎉 Ready! Run: python update_failed_to_complete.py"


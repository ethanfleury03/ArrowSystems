#!/bin/bash
# Find Cloud SQL IP and connect directly

PASSWORD="${1:-C9ruwaOM4urx1dXLI8LxD5UpXoy_ii0cjRG7xlfdZX8}"

echo "🔍 Finding Cloud SQL IP address..."
echo ""

# Try to get IP using gcloud if available
if command -v gcloud &> /dev/null; then
    echo "Using gcloud to find IP..."
    IP=$(gcloud sql instances describe arrow-rag-support-prod --format="get(ipAddresses[0].ipAddress)" 2>/dev/null || echo "")
    
    if [ -n "$IP" ] && [ "$IP" != "None" ]; then
        echo "✅ Found IP: $IP"
        echo ""
        export DATABASE_URL="postgresql://rag_user:${PASSWORD}@${IP}:5432/rag_app"
        echo "🔧 Connecting..."
        python update_failed_to_complete.py
        exit $?
    fi
fi

# If gcloud didn't work, prompt user
echo "Could not automatically find IP."
echo ""
echo "Please get the IP from Google Cloud Console:"
echo "  1. Go to: https://console.cloud.google.com/sql/instances"
echo "  2. Click on: arrow-rag-support-prod"
echo "  3. Look for 'Public IP address' or 'IP address'"
echo ""
read -p "Enter Cloud SQL IP address: " IP

if [ -z "$IP" ]; then
    echo "❌ No IP provided"
    exit 1
fi

export DATABASE_URL="postgresql://rag_user:${PASSWORD}@${IP}:5432/rag_app"
echo ""
echo "🔧 Connecting to $IP..."
python update_failed_to_complete.py


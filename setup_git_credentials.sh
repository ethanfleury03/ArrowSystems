#!/bin/bash
# One-time Git credential setup for RunPod
# Run this once and never enter credentials again

echo "=========================================="
echo "🔐 Git Credential Setup"
echo "=========================================="
echo ""
echo "Choose an option:"
echo ""
echo "1) Cache credentials (24 hours) - Temporary"
echo "2) Store credentials permanently - Stored in ~/.git-credentials"
echo "3) Use Personal Access Token (PAT) - Recommended for GitHub"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "📦 Setting up credential cache (24 hours)..."
        git config --global credential.helper 'cache --timeout=86400'
        echo "✅ Credentials will be cached for 24 hours after first entry"
        ;;
    2)
        echo ""
        echo "📦 Setting up permanent credential storage..."
        git config --global credential.helper store
        echo "✅ Credentials will be stored permanently in ~/.git-credentials"
        echo ""
        echo "⚠️  Note: Credentials are stored in plain text"
        echo "   Next git operation will prompt once, then never again"
        ;;
    3)
        echo ""
        echo "📦 Setting up GitHub Personal Access Token..."
        echo ""
        echo "Steps:"
        echo "1. Go to: https://github.com/settings/tokens"
        echo "2. Click 'Generate new token (classic)'"
        echo "3. Select scopes: repo (all)"
        echo "4. Generate and copy the token"
        echo ""
        read -p "Enter your GitHub username: " username
        read -p "Enter your Personal Access Token: " token
        
        # Store in credential store
        git config --global credential.helper store
        
        # Create credentials file with token
        echo "https://${username}:${token}@github.com" > ~/.git-credentials
        chmod 600 ~/.git-credentials
        
        echo ""
        echo "✅ PAT configured and stored"
        echo "   You won't be prompted for credentials anymore"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Test it by running:"
echo "  git pull"
echo ""


#!/bin/bash
# Start DuraFlex UI in MOCK MODE (no GPU required)

echo "========================================"
echo "DuraFlex Technical Assistant"
echo "MOCK MODE - UI Development"
echo "========================================"
echo ""
echo "Starting Streamlit in MOCK mode..."
echo "No GPU or vector index required!"
echo ""

export USE_MOCK_RAG=true
streamlit run app.py


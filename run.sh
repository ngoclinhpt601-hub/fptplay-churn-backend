#!/bin/bash

# ============================================
# FPTPlay Churn Prediction - Run Script
# ============================================

echo "Starting FPTPlay Churn Prediction..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found, using defaults"
fi

# Check if model exists
if [ ! -f "best_model_random_forest.pkl" ]; then
    echo "❌ Model file not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo "✅ Environment ready!"
echo ""
echo "========================================="
echo "Starting Flask Application..."
echo "========================================="
echo ""
echo "Server will be available at:"
echo "  ➡️ http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the application
python app.py

#!/bin/bash

# ============================================
# FPTPlay Churn Prediction - Auto Setup Script
# ============================================

set -e

echo "========================================="
echo "FPTPlay Churn Prediction Setup"
echo "========================================="
echo ""

# Step 1: Copy model file
echo "[1/5] Copying model file..."
if [ -f "../best_model_random_forest.pkl" ]; then
    cp ../best_model_random_forest.pkl .
    echo "✅ Model file copied!"
else
    echo "❌ Model file not found"
    exit 1
fi

# Step 2: Virtual environment
echo "[2/5] Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created!"
fi

# Step 3: Install dependencies
echo "[3/5] Installing dependencies..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
pip install -r requirements.txt
echo "✅ Dependencies installed!"

# Step 4: Create .env
echo "[4/5] Creating .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    sed -i.bak "s/your-super-secret-key-change-this/$SECRET_KEY/" .env
    rm .env.bak
    echo "✅ .env created!"
fi

# Step 5: Create directories
echo "[5/5] Creating directories..."
mkdir -p logs uploads
echo "✅ Setup complete!"

echo ""
echo "To start: python app.py"
echo "Then visit: http://localhost:5000"

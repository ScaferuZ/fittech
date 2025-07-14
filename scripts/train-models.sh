#!/bin/bash

# XGFitness - Model Training Script
# Run this in production if you need to retrain models

echo "🧠 XGFitness Model Training"
echo "=========================="

# Check if we're in Docker
if [ -f /.dockerenv ]; then
    echo "📦 Running inside Docker container"
    MODEL_DIR="/app/backend/models"
    BACKEND_DIR="/app/backend"
else
    echo "💻 Running on host system"
    MODEL_DIR="backend/models"
    BACKEND_DIR="backend"
fi

echo "📁 Model directory: $MODEL_DIR"

# Check if models already exist
if [ -f "$MODEL_DIR/xgfitness_ai_model.pkl" ]; then
    echo "⚠️  Trained model already exists!"
    echo "   Found: $MODEL_DIR/xgfitness_ai_model.pkl"
    
    read -p "Do you want to retrain anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Training cancelled"
        exit 0
    fi
    
    # Backup existing model
    echo "📦 Backing up existing model..."
    cp "$MODEL_DIR/xgfitness_ai_model.pkl" "$MODEL_DIR/xgfitness_ai_model_backup_$(date +%Y%m%d_%H%M%S).pkl"
fi

echo "🚀 Starting model training..."

# Navigate to backend directory and train
cd "$BACKEND_DIR"

# Train models
echo "🔄 Training models (this may take a few minutes)..."
python3 train_model.py

# Check if training was successful
if [ -f "$MODEL_DIR/xgfitness_ai_model.pkl" ]; then
    MODEL_SIZE=$(du -h "$MODEL_DIR/xgfitness_ai_model.pkl" | cut -f1)
    echo "✅ Model training completed successfully!"
    echo "📊 Model size: $MODEL_SIZE"
    echo "📍 Location: $MODEL_DIR/xgfitness_ai_model.pkl"
    
    # If in Docker, restart the service
    if [ -f /.dockerenv ]; then
        echo "🔄 Model updated in container. Consider restarting the service."
    fi
else
    echo "❌ Model training failed!"
    echo "🔍 Check the output above for errors"
    exit 1
fi

echo "🎉 Training complete!" 
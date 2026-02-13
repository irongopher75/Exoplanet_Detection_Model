#!/bin/bash

# Configuration
export MODEL_DIR="outputs/checkpoints"
export KAGGLE_USERNAME="vishnuupanicker"
export KAGGLE_KEY="b4d7088f17c1e0c8995ad7ea3a072a96"

echo "🔍 Checking for Kaggle CLI..."
# Add standard Python bin paths to PATH just in case
export PATH="$PATH:/Library/Frameworks/Python.framework/Versions/3.12/bin"

echo "🔑 Configuring Kaggle credentials..."
mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

if command -v kaggle &> /dev/null; then
    KAG_CMD="kaggle"
else
    echo "❌ Kaggle CLI not found."
    echo "💡 Try installing it with: python3 -m pip install kaggle"
    exit 1
fi

echo "✅ Using: $KAG_CMD"

echo "🚀 Starting Kaggle CLI upload process..."

# 1. Create the model repository (holding all variations)
echo "📦 Initializing model repository..."
# This may fail if already exists or due to token restrictions, 
# but if the repo is already created manually, we can still proceed.
$KAG_CMD models create -p $MODEL_DIR || echo "⚠️ Model creation skipped (already exists or permission error)"

# 2. Upload the specific model instance (variation)
echo "📤 Uploading model instance..."
$KAG_CMD models instances create -p $MODEL_DIR

echo "✅ Process complete!"

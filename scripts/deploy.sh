#!/bin/bash
# Deployment script for VoiceDoc Intelligence

set -e

echo "🚀 VoiceDoc Intelligence Deployment Script"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi

# Set project ID
read -p "Enter your Google Cloud Project ID: " PROJECT_ID
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "📦 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com

# Submit build
echo "🔨 Building and deploying to Cloud Run..."
gcloud builds submit --config=cloudbuild.yaml

echo "✅ Deployment complete!"
echo "🌐 Your application should be available at:"
gcloud run services describe voicedoc-intelligence --region=us-central1 --format='value(status.url)'

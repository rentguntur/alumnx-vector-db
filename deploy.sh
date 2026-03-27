#!/bin/bash
set -e

# Arguments
BRANCH=${1:-dev}
PORT=${2:-8001}
APP_NAME=${3:-alumnx-vector-db-dev}
REPO_DIR="/home/ubuntu/$APP_NAME"

echo "Deploying $APP_NAME (Branch: $BRANCH) on Port: $PORT..."

# Initialize directory if not exists
if [ ! -d "$REPO_DIR" ]; then
    echo "Creating directory $REPO_DIR and cloning repo..."
    git clone https://github.com/alumnx-ai-labs/alumnx-vector-db.git "$REPO_DIR"
fi

# Navigate to repo
cd "$REPO_DIR"

# Pull latest changes
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

# Standardize environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install dependencies (using uv if available, otherwise pip)
if command -v uv > /dev/null; then
    uv sync
else
    ./venv/bin/pip install -r requirements.txt
fi

# Reload or Start with PM2
if pm2 describe "$APP_NAME" > /dev/null 2>&1; then
    echo "Reloading $APP_NAME..."
    pm2 reload "$APP_NAME"
else
    echo "Starting $APP_NAME for the first time..."
    pm2 start "uv run uvicorn main:app --host 0.0.0.0 --port $PORT" --name "$APP_NAME"
fi

# Save pm2 state
pm2 save

echo "Deployment of $APP_NAME completed successfully!"

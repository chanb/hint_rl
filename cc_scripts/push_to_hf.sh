#!/bin/bash
# push_to_hf.sh - Upload a trained checkpoint to HuggingFace Hub
#
# Usage:
#   ./push_to_hf.sh <checkpoint_dir> [<repo_id>]
#
# Examples:
#   # Push to default repo (FengdiFlo/HintRL)
#   ./push_to_hf.sh /scratch/hint_rl_results/checkpoints/.../epoch10epochstep13globalstep153
#
#   # Push to a specific repo
#   ./push_to_hf.sh /scratch/hint_rl_results/checkpoints/.../epoch10epochstep13globalstep153 FengdiFlo/HintRL
#
# Environment:
#   HF_TOKEN - HuggingFace token (set below or via environment)
#   Activate the project venv first: source .venv/bin/activate
#
# Token is scoped to FengdiFlo/HintRL with read+write permissions.

set -euo pipefail

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir> [<repo_id>]}"
REPO_ID="${2:-FengdiFlo/HintRL}"

# Load token from .env if not already set
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi
: "${HF_TOKEN:?HF_TOKEN not set. Add it to .env or export it.}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

if [ ! -f "$CHECKPOINT_DIR/config.json" ]; then
    echo "Warning: config.json not found in $CHECKPOINT_DIR — may not be an HF-format checkpoint"
fi

echo "Uploading checkpoint to HuggingFace Hub..."
echo "  Source: $CHECKPOINT_DIR"
echo "  Repo:   $REPO_ID"

huggingface-cli upload "$REPO_ID" "$CHECKPOINT_DIR" --token "$HF_TOKEN"

echo "Done. Model available at: https://huggingface.co/$REPO_ID"

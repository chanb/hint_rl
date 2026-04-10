#!/bin/bash
# pull_from_hf.sh - Download a model from HuggingFace Hub to a local directory
#
# Usage:
#   ./pull_from_hf.sh <repo_id> <local_dir>
#
# Examples:
#   # Download base model
#   ./pull_from_hf.sh nvidia/OpenMath-Nemotron-1.5B /home/fengdic/scratch/models/OpenMath-Nemotron-1.5B
#
#   # Download our trained checkpoint
#   ./pull_from_hf.sh FengdiFlo/HintRL /home/fengdic/scratch/models/HintRL
#
# Environment:
#   HF_TOKEN - HuggingFace token (set below or via environment)
#   Activate the project venv first: source .venv/bin/activate

set -euo pipefail

REPO_ID="${1:?Usage: $0 <repo_id> <local_dir>}"
LOCAL_DIR="${2:?Usage: $0 <repo_id> <local_dir>}"

# Load token from .env if not already set
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi
: "${HF_TOKEN:?HF_TOKEN not set. Add it to .env or export it.}"

echo "Downloading model from HuggingFace Hub..."
echo "  Repo:   $REPO_ID"
echo "  Dest:   $LOCAL_DIR"

huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR" --token "$HF_TOKEN"

echo "Done. Model saved to: $LOCAL_DIR"
echo ""
echo "To use in evaluation:"
echo "  actor.path=$LOCAL_DIR"

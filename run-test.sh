#!/bin/bash
# Test script for alt-text generator

# Load environment variables from .env.local
if [ -f .env.local ]; then
    set -a
    source .env.local
    set +a
else
    echo "ERROR: .env.local file not found"
    echo "Create .env.local with your credentials (see README.md)"
    exit 1
fi

# Run dry-run test with 5 images (quality-first model chain)
./altgen --dry --limit-images=5 --report=test.jsonl --examples=examples.json

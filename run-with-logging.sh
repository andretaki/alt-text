#!/bin/bash
# Run with full logging to both screen and file

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

# Create log filename with timestamp
LOGFILE="run-$(date +%Y%m%d-%H%M%S).log"

echo "Starting run... Logging to $LOGFILE"
echo "View live: tail -f $LOGFILE"
echo ""

# Run with tee to show on screen AND save to log file
./altgen --dry --limit-images=5 --report=test.jsonl --examples=examples.json 2>&1 | tee "$LOGFILE"

echo ""
echo "Run complete. Log saved to: $LOGFILE"
echo "Report saved to: test.jsonl"

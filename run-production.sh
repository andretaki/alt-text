#!/bin/bash
# Production run with full logging - UPLOADS TO SHOPIFY

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

# Create filenames with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="production-$TIMESTAMP.log"
REPORTFILE="production-$TIMESTAMP.jsonl"

echo "=========================================="
echo "  PRODUCTION RUN - WILL UPLOAD TO SHOPIFY"
echo "=========================================="
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo ""
echo "Starting production run..."
echo "Log file: $LOGFILE"
echo "Report file: $REPORTFILE"
echo ""
echo "To monitor in another terminal: tail -f $LOGFILE"
echo ""

# Run with tee to show on screen AND save to log file
./altgen --report="$REPORTFILE" --examples=examples.json 2>&1 | tee "$LOGFILE"

echo ""
echo "=========================================="
echo "  PRODUCTION RUN COMPLETE"
echo "=========================================="
echo "Log saved to: $LOGFILE"
echo "Report saved to: $REPORTFILE"
echo ""
echo "Quick stats:"
echo "  Total processed: $(grep -c 'Generated alt text' "$LOGFILE" || echo 0)"
echo "  Skipped (existing): $(grep 'skipped_existing=' "$LOGFILE" | tail -1 | grep -oP 'skipped_existing=\K[0-9]+' || echo 0)"
echo ""
echo "Review report: jq . $REPORTFILE | less"

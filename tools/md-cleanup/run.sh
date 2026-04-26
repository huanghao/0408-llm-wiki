#!/bin/bash
# Usage: ./run.sh <input.md> [output.md]
# If output is omitted, prints to stdout.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$1" ]; then
    echo "Usage: $0 <input.md> [output.md]" >&2
    exit 1
fi

if [ -n "$2" ]; then
    python3 "$SCRIPT_DIR/cleanup.py" "$1" -o "$2" -v
else
    python3 "$SCRIPT_DIR/cleanup.py" "$1" -v
fi

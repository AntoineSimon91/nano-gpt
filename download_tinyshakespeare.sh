#!/bin/bash
# This script downloads the Tiny Shakespeare dataset using wget

# URL of the file to download
URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Destination directory and file path
DEST_DIR="data"
DEST_FILE="$DEST_DIR/input.txt"

# Check if the data directory exists, if not, create it
if [ ! -d "$DEST_DIR" ]; then
  echo "Directory $DEST_DIR does not exist. Creating it now."
  mkdir -p "$DEST_DIR"
fi

# Download the file
wget -O "$DEST_FILE" "$URL"

echo "Download complete. File saved as $DEST_FILE"
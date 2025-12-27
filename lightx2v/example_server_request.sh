#!/bin/bash

# Example script to send a request to the Image Edit Server working on localhost:8000
# Usage: ./example_server_request.sh [image_path] [prompt]

IMAGE_PATH="${1:-/workspace/eagle_input.jpg}"
PROMPT="${2:-add eagle}"
SEED="${3:-42}"

echo "Sending request to server..."
echo "Image: $IMAGE_PATH"
echo "Prompt: $PROMPT"

# Use curl to send POST request
curl -X POST "http://localhost:8000/edit" \
     -H "Content-Type: application/json" \
     -d "{
           \"images\": [\"$IMAGE_PATH\"],
           \"prompt\": \"$PROMPT\",
           \"seed\": $SEED
         }" | python3 -m json.tool

echo -e "\nRequest sent!"

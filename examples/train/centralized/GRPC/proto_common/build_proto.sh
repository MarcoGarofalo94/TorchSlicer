#!/bin/bash

# Check if a parameter has been passed
if [ -z "$1" ]; then
  echo "Usage: $0 <proto_file>"
  exit 1
fi

PROTO_FILE="$1"

# Run the protoc command with the provided parameter
python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. "$PROTO_FILE"

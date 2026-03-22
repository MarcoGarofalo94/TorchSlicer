#!/bin/bash
set -e

# Recompile proto files from the library root so gencode/runtime versions always match.
# The library is installed at /usr/local/lib/torchslicer (editable install via volume mount).
python3 -m grpc_tools.protoc \
    -I /usr/local/lib/torchslicer \
    --python_out=/usr/local/lib/torchslicer \
    --grpc_python_out=/usr/local/lib/torchslicer \
    torchslicer/transport/grpc/coordinator/coordinator_service.proto \
    torchslicer/transport/grpc/worker/worker_service.proto

exec "$@"

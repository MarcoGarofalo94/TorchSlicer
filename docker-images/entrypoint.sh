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

# PyTorch cache-dir setup calls getpass.getuser() → pwd.getpwuid(uid) which fails
# when the container runs as a numeric UID not present in /etc/passwd.
# Setting LOGNAME makes getpass use the env var path instead.
export LOGNAME=${LOGNAME:-user}
export HOME=${HOME:-/tmp}

exec "$@"

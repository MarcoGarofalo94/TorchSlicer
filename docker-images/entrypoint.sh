#!/bin/bash
set -e

# Recompile proto files so gencode/runtime versions always match.
#
# Why here AND in the Dockerfile:
#   Production images (no volume mount) use the protos compiled at build time.
#   Dev images mount ./lib/torchslicer over /usr/local/lib/torchslicer, which
#   replaces the Dockerfile-compiled files with the local source tree.
#   This compile step regenerates them against the installed grpcio version.
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

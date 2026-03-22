#!/bin/bash
set -e

# Regenerate all .proto files found under /workspace so the gencode version
# always matches the installed grpcio/protobuf runtime.
find /workspace -name "*.proto" | while read -r proto; do
    root=$(dirname "$proto")
    # Walk up to find the directory that contains proto_common/
    while [ "$root" != "/" ] && [ ! -d "$root/proto_common" ]; do
        root=$(dirname "$root")
    done
    if [ -d "$root/proto_common" ]; then
        python3 -m grpc_tools.protoc -I"$root" \
            --python_out="$root" --grpc_python_out="$root" "$proto" 2>/dev/null || true
    fi
done

exec "$@"

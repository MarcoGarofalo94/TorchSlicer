import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proto_common.coordinator import coordinator_service_pb2
from proto_common.coordinator import coordinator_service_pb2_grpc

print(coordinator_service_pb2.Empty)


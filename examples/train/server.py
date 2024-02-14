import grpc
import tensor_service_pb2
import tensor_service_pb2_grpc
import numpy as np
import torch
from concurrent import futures

class TensorServicer(tensor_service_pb2_grpc.TensorServiceServicer):
    def forward(self, request, context):
        # Deserialize the received tensor
        tensor_np = np.frombuffer(request.data, dtype=np.float32)
        tensor = torch.tensor(tensor_np)

        # Process the tensor (for example, calculate its sum)
        result = tensor.sum()

        # Serialize the result tensor
        result_data = result.numpy().tobytes()

        # Return the result tensor
        return tensor_service_pb2.TensorMessage(data=result_data)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tensor_service_pb2_grpc.add_TensorServiceServicer_to_server(
        TensorServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()

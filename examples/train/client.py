import grpc
import tensor_service_pb2
import tensor_service_pb2_grpc
import numpy as np
import torch

def run():
    # Create a random tensor
    tensor = torch.randn(3, 3)

    # Serialize the tensor
    tensor_data = tensor.numpy().tobytes()

    # Create gRPC channel
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = tensor_service_pb2_grpc.TensorServiceStub(channel)

        # Create TensorMessage
        tensor_message = tensor_service_pb2.TensorMessage(data=tensor_data)
        tensor_message.writeDelimitedTo(open('input_tensor.pt', 'wb'))

        # Call RPC and get response
        response = stub.ProcessTensor(tensor_message)

        # Deserialize the response tensor
        result_tensor_np = np.frombuffer(response.data, dtype=np.float32)
        result_tensor = torch.tensor(result_tensor_np)

    print("Received result tensor:", result_tensor)

if __name__ == '__main__':
    run()

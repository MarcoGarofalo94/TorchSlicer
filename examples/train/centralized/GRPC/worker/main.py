from collections import OrderedDict
import threading
import grpc
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from proto_common.coordinator import coordinator_service_pb2_grpc
from proto_common.coordinator import coordinator_service_pb2
from proto_common.worker import worker_service_pb2_grpc
from proto_common.worker import worker_service_pb2

import numpy as np
import torch
from torch import nn
from torch import optim
from concurrent import futures
from torchslicer.SplitLayer import SplitLayer
from torchslicer.TorchSlicer import Slicer
from google.protobuf import json_format
import io
import socket
import asyncio

import datetime
import base64



class WorkerServicer(worker_service_pb2_grpc.WorkerServiceServicer):

    def __init__(self):
        super().__init__()
        self.layer: SplitLayer = None
        self.loss_fn: torch.nn.functional = None
        self.loss: torch.Tensor = None
        self.prev_layer: str = None
        self.next_layer: str = None
        self.out_tensor: torch.Tensor = None
        self.label: torch.Tensor = None
        self.debug: bool = False

    def init(self, request, context):

        # print(request)
        try:
            config = json_format.MessageToDict(request)
            # print(config)
            raw_layers = config['layers']
            optimizer = config['optimizer']
            init_state_dict = config['initStateDict'] if request.HasField(
                'init_state_dict') else False
            is_last = config['isLast'] if request.HasField(
                'is_last') else False
            if is_last:
                self.loss_fn = getattr(nn, config['criterion']['name'])(
                    **config['criterion']['params'])

            self.prev_layer = config['prevLayer'] if request.HasField(
                'prev_layer') else None
            self.next_layer = config['nextLayer'] if request.HasField(
                'next_layer') else None

            slices = Slicer.cast_layers(raw_layers)
            
            torch_layers = self.slice_to_torch(slices)

            self.layer = SplitLayer(torch_layers, is_last)
            if init_state_dict:
                res = self._load_state_dict(slices)
            optimizer = getattr(optim, optimizer['name'])(self.layer.parameters(),
                                                          **optimizer['params'])
            self.layer.set_optimizer(optimizer)
            print('SPLITLAYER', self.layer)
            print('OPTIMIZER', optimizer)
            print('LOSS', self.loss_fn)
            return worker_service_pb2.LogMessage(message="Layer Initialized", hostname=socket.gethostname(), ip_addr=socket.gethostbyname(socket.gethostname()))
        except Exception as e:
            print(e)
            return worker_service_pb2.LogMessage(message="Could not initialize layer - "+str(e), hostname=socket.gethostname(), ip_addr=socket.gethostbyname(socket.gethostname()))

    def set_label(self, request, context):
        
        try:
            self.label = self._deserialize_tensor(request.data)
            self.debug and print(self.label)
            return worker_service_pb2.LogMessage(message="Label Set", hostname=socket.gethostname(), ip_addr=socket.gethostbyname(socket.gethostname()))
        except Exception as e:
            return worker_service_pb2.LogMessage(message="Could not set label - "+str(e), hostname=socket.gethostname(), ip_addr=socket.gethostbyname(socket.gethostname()))

    def forward(self, request: worker_service_pb2.ForwardMessage, context: grpc.ServicerContext):
        
        threading.Thread(target=self.__forward__, args=(request, context)).start()
        return worker_service_pb2.ForwardStatusMessage(status="1", message="Forwarding to next layer")

        # self.debug and print('[FORWARD START] - ', sys.argv[1], " - ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # # await context.send_initial_metadata([('key', 'value')])
        # # await context.write(tensor_service_pb2.TensorMessage(tensor=tensor_service_pb2.Tensor(data=b'1231')))
        # # Deserialize the received tensor
        # tensor = self._deserialize_tensor(request.input.data)
        # self.debug and print('[FORWARD input deserialized] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # # forward pass of the layer
        # print(tensor.shape)
        # self.out_tensor = self._forward(tensor)
        # self.debug and print('[FORWARD output calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # # print(self.out_tensor, self.layer._is_last)
        # if self.layer._is_last:  # this layer is last layer
        #     self.debug and print('[FORWARD last layer] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )

        #     # starting backward pass

        #     if self.label != None:  # check that tensor label is present
        #         self.loss = self.loss_fn(self.out_tensor, self.label)
        #         self.debug and print('[FORWARD loss calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )

        #         self.backward(None,None)
        #         self.layer.optimize()
        # else:
        #     print(self.next_layer)
        #     ch = grpc.insecure_channel(self.next_layer)
        #     stub = tensor_service_pb2_grpc.TensorServiceStub(ch)

        # # Serialize the result tensor
        #     buff = io.BytesIO()
        #     torch.save(self.out_tensor, buff)
        #     result_data = buff.getvalue()
        #     stub.forward(tensor_service_pb2.ForwardMessage(
        #         input=tensor_service_pb2.Tensor(data=result_data)))
            # Return the result tensor
        #return tensor_service_pb2.TensorMessage(tensor=tensor_service_pb2.Tensor(data=))
        # return tensor_service_pb2.TensorMessage(tensor=tensor_service_pb2.Tensor(data=result_data))

    def backward(self, request, context):
        threading.Thread(target=self.__backward__, args=(request, context)).start()
        return worker_service_pb2.BackwardStatusMessage(status="1", message="Backwarding to previous layer")
        # self.debug and print('[BACKWARD START] - ', sys.argv[1], " - ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
   
        # if self.loss != None:
        #     grad = self.layer.backward(loss=self.loss)
        #     self.debug and print('[BACKWARD loss calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # else:
        #     tensor = self._deserialize_tensor(request.gradient.data)
        #     self.debug and print('[BACKWARD gradient deserialized] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        #     # print(tensor)
        #     grad = self._backward(prev_g=tensor,out=self.out_tensor)
        #     self.debug and print('[BACKWARD gradient calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        #     # print(grad)

        # if self.prev_layer != None:
        #     self._prev_backward(
        #         prev_layer=self.prev_layer, prev_g=grad)
        #     self.debug and print('[BACKWARD prev layer] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
   
    def __forward__(self, request, context):
        self.debug and print('[FORWARD START] - ', sys.argv[1], " - ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # await context.send_initial_metadata([('key', 'value')])
        # await context.write(tensor_service_pb2.TensorMessage(tensor=tensor_service_pb2.Tensor(data=b'1231')))
        # Deserialize the received tensor
        tensor = self._deserialize_tensor(request.input.data)
        self.debug and print('[FORWARD input deserialized] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # forward pass of the layer
        self.debug and print(tensor.shape)
        self.out_tensor = self._forward(tensor)
        self.debug and print('[FORWARD output calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # print(self.out_tensor, self.layer._is_last)
        if self.layer._is_last:  # this layer is last layer
            self.debug and print('[FORWARD last layer] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )

            # starting backward pass

            if self.label != None:  # check that tensor label is present
                self.loss = self.loss_fn(self.out_tensor, self.label)
                self.debug and print("LOSS", self.loss)
                channel = grpc.insecure_channel('localhost:50054')
                stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(channel)
                coordinator_service_pb2.Loss
                stub.show_loss(coordinator_service_pb2.LossMessage(loss=self.loss.item()))

                self.debug and print('[FORWARD loss calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )

                self.backward(None,None)
                
        else:
            self.debug and print(self.next_layer)
            channel = grpc.insecure_channel(self.next_layer)
            stub = worker_service_pb2_grpc.WorkerServiceStub(channel)

        # Serialize the result tensor
            buff = io.BytesIO()
            torch.save(self.out_tensor, buff)
            result_data = buff.getvalue()
            buff.close()
            stub.forward(worker_service_pb2.ForwardMessage(
                input=worker_service_pb2.Tensor(data=result_data)))
            # channel.close()
            
    def __backward__(self, request, context):
        self.debug and print('[BACKWARD START] - ', sys.argv[1], " - ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
   
        if self.loss != None:
            grad = self.layer.backward(loss=self.loss)
            self.debug and print('[BACKWARD loss calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        else:
            tensor = self._deserialize_tensor(request.gradient.data)
            self.debug and print('[BACKWARD gradient deserialized] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
            # print(tensor)
            grad = self._backward(prev_g=tensor,out=self.out_tensor)
            self.debug and print('[BACKWARD gradient calculated] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
            # print(grad)
        self.layer.optimize()
        if self.prev_layer != None:
            self._prev_backward(
                prev_layer=self.prev_layer, prev_g=grad)
            self.debug and print('[BACKWARD prev layer] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        else:
            channel = grpc.insecure_channel('localhost:50054')
            stub = coordinator_service_pb2_grpc.CoordinatorServiceStub(channel)
            stub.batch_done(coordinator_service_pb2.Empty())
            # channel.close()


    def _forward(self, input_tensor):

        return self.layer(input_tensor)

    def _backward(self, prev_g=None, loss=None, out=None):

        return self.layer.backward(prev_g=prev_g, loss=loss, out=out)

    def _prev_backward(self, prev_layer, prev_g):
        channel = grpc.insecure_channel(prev_layer)
        stub = worker_service_pb2_grpc.WorkerServiceStub(channel)
        self.debug and print('[BACKWARD prev CHANNEL CREATED] - ',prev_layer," - ", sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        grad_buff = io.BytesIO()
        torch.save(prev_g, grad_buff)
        grad_data = grad_buff.getvalue()
        grad_buff.close()
        self.debug and print('[BACKWARD prev grad serialized] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
            # print(grad_data)
        grad_tensor = worker_service_pb2.Tensor(data=grad_data)
        # print(grad_tensor)
        response = stub.backward(
                worker_service_pb2.BackwardMessage(gradient=grad_tensor))
        self.debug and print('[BACKWARD prev stub sent] - ', sys.argv[1],datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] )
        # channel.close()
        

    def _optimize(self,):
        return self.layer.optimize(dry_run=False)

    def _deserialize_tensor(self, tensor_data: bytes) -> torch.Tensor:
        return torch.load(io.BytesIO(tensor_data))

    def slice_to_torch(self, slices):

        layers = []
        for slice in slices:
            layers.append(getattr(nn, slice['layer'])(
                **slice['decoded_attributes']))
        return layers
    
    def _load_state_dict(self, slices):
        layer_keys = list(self.layer.layers.state_dict().keys())
        group_list = {}
        for k in layer_keys:
            layer, param = k.split(".")
            layer = int(layer)
            if layer not in group_list:
                group_list[layer] = [k]
            else:
                group_list[layer].append(k)
        state_dict = OrderedDict()

        for i, slice in enumerate(slices):
            if "state_dict" in slice:
                sd_buff = io.BytesIO(base64.b64decode(slice["state_dict"]))
                sd = torch.load(sd_buff)
                sd_buff.close()
                values = list(sd.values())
                
                for param_key, param_value in zip(group_list[i], values):
                    state_dict.update({param_key: param_value})
        return self.layer.layers.load_state_dict(state_dict)
    



def serve():
    port = sys.argv[1]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    worker_service_pb2_grpc.add_WorkerServiceServicer_to_server(
        WorkerServicer(), server)
    server.add_insecure_port('[::]:'+port)
    print("Server started at port "+port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
   serve()

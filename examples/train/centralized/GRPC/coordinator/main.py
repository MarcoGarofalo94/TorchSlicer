import asyncio
from collections import OrderedDict
import io

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import grpc
from proto_common.coordinator import coordinator_service_pb2_grpc
from proto_common.coordinator import coordinator_service_pb2
from proto_common.worker import worker_service_pb2_grpc
from proto_common.worker import worker_service_pb2

# import centralized.GRPC.proto_common.coordinator.coordinator_service_pb2 as coordinator_service_pb2
# import centralized.GRPC.proto_common.coordinator.coordinator_service_pb2_grpc as coordinator_service_pb2_grpc

# import centralized.GRPC.proto_common.worker.worker_service_pb2 as worker_service_pb2
# import centralized.GRPC.proto_common.worker.worker_service_pb2_grpc as worker_service_pb2_grpc

import numpy as np
import torch
from torch import nn
from torchslicer.TorchSlicer import Slicer
from google.protobuf.struct_pb2 import Struct
# from kubernetes import client, config
import math
import pprint
from google.protobuf import json_format
import threading
from concurrent import futures
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import typing
import time
import yaml
from torch.utils.data import DataLoader, random_split, TensorDataset
import gzip
import requests

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


model = SimpleCNN()
# model.conv1.state_dict()
# print(model.conv1.state_dict())
# exit()
net = Slicer(model)
slices = net.get_config(init_state_dict=True)
# ml = nn.ModuleList([
#     nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(
#         1, 1), padding=(1, 1), dtype=torch.float32),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
#                  dilation=1, ceil_mode=False),
#     nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#     nn.ReLU(),
#     nn.MaxPool2d(kernel_size=2, stride=2, padding=0,
#                  dilation=1, ceil_mode=False),
#     # nn.Flatten(start_dim=1, end_dim=-1),
#     # nn.Linear(in_features=3136, out_features=128, bias=True),
#     # nn.ReLU(),
#     # nn.Linear(in_features=128, out_features=10, bias=True)
# ])

# ml_keys = list(ml.state_dict().keys())
# group_list = {}
# for k in ml_keys:
#     layer, param = k.split(".")
#     if layer not in group_list:
#         group_list[layer] = [k]
#     else:
#         group_list[layer].append(k)
# keys = list(group_list.values())
# a = OrderedDict()
# print(keys)

# k = 0
# for i, slice in enumerate(slices):
#     if "state_dict" in slice and k < len(keys):
#         sd_buff = io.BytesIO(slice["state_dict"])
#         sd = torch.load(sd_buff)
#         sd_buff.close()
#         values = list(sd.values())

#         # print(len(keys[k]),len(values))

#         for j in range(len(keys[k])):
#             print(k,j)
#             a.update({keys[k][j]: values[j]})
#         k += 1
# print(ml.load_state_dict(a))

# group_list.append(k.split(".")[0])
# a = OrderedDict()
# for s in slices:
#     if "state_dict" in s:


# print(a)

# exit()
# print(slices)
# exit()

# config.load_kube_config(config_file="config")
# v1 = client.CoreV1Api()
# cluster = v1.list_node().items

# item_n = 0
# associated_nodes = []
# l_per_dev = math.ceil(len(slices)/len(v1.list_node().items)) + 1
# for i in range(0, len(slices), l_per_dev):

#     if i+l_per_dev > len(slices):
#         associated_nodes.append(
#             {cluster[item_n].metadata.name: [s for s in slices[i:]]})
#     else:
#         associated_nodes.append({cluster[item_n].metadata.name: [
#                                 s for s in slices[i:i+l_per_dev]]})
#         # print(slices[i:i+l_per_dev])
#         # print("Node: ", cluster[item_n].metadata.name)
#         item_n += (1 % len(cluster))
#         # print("----")

# pprint.pprint(associated_nodes)

# Create gRPC channel
# with grpc.insecure_channel('localhost:50052') as channel:
#     stub = tensor_service_pb2_grpc.TensorServiceStub(channel)
#     struct_slices = Struct()
#     # struct_slices.update({slices})
#     # print(struct_slices)
#     # Create TensorMessage
#     import json
#     struct_slices = []
#     for slice in slices:
#         s = Struct()
#         s.update(slice)
#         struct_slices.append(s)
#     print(struct_slices)
#     config_message = tensor_service_pb2.Config(layers=struct_slices)
#     layers_dict = json_format.MessageToDict(config_message)
#     print(layers_dict)
#     Slicer.cast_layers(layers_dict=layers_dict)
# for i, layer in enumerate(layers_dict['layers']):
#     layers_dict['layers'][i]['new_attr'] = {}
#     for attr, val in layer['attributes'].items():
#         if (val['param_type'] not in ['float', 'dtype', "int"]) and (not isinstance(val['param_type'], (list, tuple))):
#             continue
#         if val['param_type'] == 'dtype':
#             layers_dict['layers'][i]['attributes'][attr]['value'] = eval(
#                 val["value"])
#         elif isinstance(val['param_type'], list):
#             layers_dict['layers'][i]['attributes'][attr]['value'] = [
#                 eval(val["param_type"][j]+"("+str(v)+")") for j, v in enumerate(val['value'])]
#         else:
#             layers_dict['layers'][i]['attributes'][attr]['value'] = eval(
#                 val["param_type"]+"("+str(val['value'])+")")
#         layers_dict['layers'][i]['new_attr'][attr] = layers_dict['layers'][i]['attributes'][attr]['value']
# print(layers_dict['layers'])

# for i, slice in enumerate(layers_dict['layers']):
#     print(getattr(nn, slice['layer'])(**slice['decoded_attributes']))

#     print(i,slice["layer"])
#     print(slice['attributes'])

#     for key,value in slice['attributes'].items():
#         if key == 'dtype':
#           slice['attributes'][key] = eval(slice['attributes'][key])

#         if isinstance(value,(list,tuple)):
#             slice['attributes'][key] = tuple(int(v) for v in value)
# a = eval('typing.Union[int, typing.Tuple[int, int]]')
# a.__args__ = (int, tuple)

#     print(slice['attributes'])
#     print(getattr(nn, slice['layer'])(**slice['attributes']))


# l = nn.Linear(in_features=3, out_features=3)
# p = {'dilation': (1.0, 1.0),
#      'dtype': eval('torch.float32'),
#      'groups': 1.0,
#      'in_channels': 32.0,
#      'kernel_size': (3.0, 3.0),
#      'out_channels': 64.0,
#      'padding': (1.0, 1.0),
#      'padding_mode': 'zeros',
#      'stride': (1.0, 1.0)}
# # lp = nn.Conv2d(in_channels=p['in_channels'], out_channels=p['out_channels'],
# #                kernel_size=p['kernel_size'], stride=p['stride'],
# #                padding=p['padding'], dtype=eval(p['dtype']),
# #                groups=p['groups'], padding_mode=p['padding_mode'])
# lp = getattr(nn, 'Conv2d')(**p)
# # Call RPC and get response
#     response = stub.init(config_message)
#     print(response)
#     # Deserialize the response tensor
#     result_tensor_np = np.frombuffer(
#         response.input_tensor.data, dtype=np.float32)
#     result_tensor = torch.tensor(result_tensor_np, dtype=eval(
#         response.input_tensor.dtype)).reshape(list(response.input_tensor.shape))
# print("Sent tensor:", tensor.data, tensor.shape)
# print("Received result tensor:", result_tensor, result_tensor.shape)


# input_choice = ""
# train_loader = torch.load('trainloader.pth')
# while input_choice != "0":
#     input_choice = input(
#         "Enter 1 to init, 2 to forward, 3 to backward, 0 to exit: ")
#     stub1 = tensor_service_pb2_grpc.TensorServiceStub(channel1)
#     stub2 = tensor_service_pb2_grpc.TensorServiceStub(channel2)
#     if input_choice == "1":
#         struct_slices = Struct()
#         # struct_slices.update({slices})
#         # print(struct_slices)
#         # Create TensorMessage
#         struct_slices = []
#         for slice in slices:
#             s = Struct()
#             s.update(slice)
#             struct_slices.append(s)
#         # print(struct_slices)
#         optimizer = Struct()
#         optimizer.update({"name": "Adam", "params": {
#             "lr": 0.001, "weight_decay": 0.0001}})
#         criterion = Struct()
#         criterion.update({"name": "CrossEntropyLoss", "params": {}})
#         cut = 7
#         print(len(struct_slices), cut, len(
#             struct_slices[0:cut]), len(struct_slices[cut:]))
#         # print(struct_slices[0:cut], struct_slices[cut:])
#         print(slices[0:cut], slices[cut:])

#         config_message1 = tensor_service_pb2.Config(
#             layers=struct_slices[0:cut], is_last=False, optimizer=optimizer, next_layer="localhost:50053", )
#         # print(config_message1)
#         config_message2 = tensor_service_pb2.Config(
#             layers=struct_slices[cut:], is_last=True, optimizer=optimizer, criterion=criterion,  prev_layer="localhost:50052")
#         # print(config_message2)
#         # Call RPC and get response
#         response1 = stub1.init(config_message1)
#         print(response1)
#         response2 = stub2.init(config_message2)
#         print(response2)
#     elif input_choice == "2":
#         inputs, label = next(iter(train_loader))

#         buff_label = io.BytesIO()
#         torch.save(label, buff_label)
#         label_tensor_data = buff_label.getvalue()
#         label_tensor = tensor_service_pb2.Tensor(
#             data=label_tensor_data)
#         label_response = stub2.set_label(label_tensor)

#         print(label_response)
#         # # tensor_data = inputs.numpy().tobytes()
#         buff_input = io.BytesIO()
#         torch.save(inputs, buff_input)
#         tensor_data = buff_input.getvalue()
#         input_tensor = tensor_service_pb2.Tensor(
#             data=tensor_data)

#         # print(input_tensor)
#         stub1.forward(
#             tensor_service_pb2.ForwardMessage(input=input_tensor))

#     elif input_choice == "3":
#         response = stub.backward(tensor_service_pb2.BackwardMessage())
#         print(response)
#     else:
#         break

"""
    train_loader = torch.load('trainloader.pth')
    inputs, label = next(iter(train_loader))
    # # tensor_data = inputs.numpy().tobytes()
    buff_input = io.BytesIO()
    torch.save(inputs, buff_input)
    tensor_data = buff_input.getvalue()

    buff_label = io.BytesIO()
    torch.save(label, buff_label)
    label_tensor_data = buff_label.getvalue()

    label_tensor = tensor_service_pb2.Tensor(
        data=label_tensor_data)

    input_tensor = tensor_service_pb2.Tensor(
        data=tensor_data)

    # print(input_tensor)
    response = stub.forward(
        tensor_service_pb2.ForwardMessage(input=input_tensor, label=label_tensor))
    # print(inputs.shape)
    # print(torch.equal(torch.load(io.BytesIO(input_tensor.data)), inputs))
"""
# reconstructed_input = torch.load(io.BytesIO(input_tensor.data))
# print(torch.equal(reconstructed_input,inputs))



class Device():
    def __init__(self, name="", address="", port=""):
        self.name = name
        self.address = address
        self.port = port
        self.stub = None

    def __str__(self) -> str:
        return f"Device: {self.name}\nAddress: {self.address} \nPort: {self.port}\n"

    def __repr__(self) -> str:
        return f"Device: {self.name}\nAddress: {self.address} \nPort: {self.port}\n"

    def get_url(self):
        return f"{self.address}:{self.port}"
    
    def get_stub(self):
        return self.stub
    
    def set_stub(self):
        self.stub = worker_service_pb2_grpc.WorkerServiceStub(grpc.insecure_channel(self.get_url()))

    def close_stub(self):
        self.stub.close()


class Dispatcher(coordinator_service_pb2_grpc.CoordinatorServiceServicer):
    """
    This class is used to dispatch the training of a neural network to multiple devices in a sequential manner.
    """

    def __init__(self, network, criterion, optimizer, max_epochs=1, gen_dataset=None, cluster=[], init_state_dict=True):
        super().__init__()

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        # get the dataset i.e. populate train, val and test dataset
        self._get_dataset(gen_dataset=gen_dataset)
        # initialize epochs and batches to 0
        self.current_epoch = 0
        self.max_epochs = max_epochs
        self.current_batch = 0
        self.cluster = cluster 

        # represent the network
        self.slicer = Slicer(network)
        # get the configuration of the network
        self.slices = self.slicer.get_config(init_state_dict=init_state_dict)

        
        struct_slices = []
        for slice in self.slices:
            struct_slices.append(self._create_struct(slice))
        _optimizer = self._set_optimizer(optimizer)
        _criterion = self._set_criterion(criterion)

        self._set_criterion(criterion)
        # logic to split the network into parts based on heuristics
        cut = 7
        # print(len(struct_slices), cut, len(
        #     struct_slices[0:cut]), len(struct_slices[cut:]))
        # print(struct_slices[0:cut], struct_slices[cut:])
        # print(slices[0:cut], slices[cut:])

        
        configs = [worker_service_pb2.Config(
            layers=struct_slices[0:cut], is_last=False, optimizer=_optimizer, next_layer="localhost:50053", init_state_dict=init_state_dict),
            worker_service_pb2.Config(
            layers=struct_slices[cut:], is_last=True, optimizer=_optimizer, criterion=_criterion,  prev_layer="localhost:50052", init_state_dict=init_state_dict)
            ]
        for dev in cluster:
            dev.set_stub()
        
        for config, dev in zip(configs, cluster):
            res = dev.get_stub().init(config)
            print(res)
        self.init_time = time.time()
        self.batch_done(None, None)

    # def _set_config(self, configs):
    #     for dev in self.cluster:
    #         channel = self._create_channel(dev)
    #         stub = tensor_service_pb2_grpc.TensorServiceStub(channel)
    #         response = stub.init(configs[dev])
    #         print(response)
    #         channel.close()

    def _create_channel(self, dev):
        return grpc.insecure_channel(dev)
    
    def _create_struct(self, value):
        s = Struct()
        s.update(value)
        return s
    
    def _set_optimizer(self, optimizer):
        return self._create_struct(optimizer)

    def _set_criterion(self, criterion):
        return self._create_struct(criterion)
    
    def show_loss(self, request, context):
        print("LOSS",request)
        return coordinator_service_pb2.Empty()

    def batch_done(self, request, context):
        
        try:
            if self.current_epoch == self.max_epochs:
                self.end_time = time.time()
                
                print('Train FINISHED! ', self.end_time - self.init_time)
                return coordinator_service_pb2.Empty()

            if self.current_batch < len(self.train_loader) - 1:
                self.current_batch += 1
                print("epoch: ", self.current_epoch , " batch: ", self.current_batch)
                
                data = next(iter(self.train_loader))
                # index, data = data
                inputs, label = data
                # print('re',inputs)
                threading.Thread(target=self.__next_epoch__,
                                 args=(inputs, label)).start()
            else:
                print('new epoch')
                self.current_batch = 0
                self.current_epoch += 1
                self.train_loader = torch.load('trainloader.pth')
                self.data = enumerate(self.train_loader)
                data = next(iter(self.data))
                index, data = data
                inputs, label = data
                # print('re2',inputs,label)

                threading.Thread(target=self.__next_epoch__,
                                 args=(inputs, label)).start()
        except Exception as e:
            print(e)
        return coordinator_service_pb2.Empty()

    def __next_epoch__(self, inputs, label):
        # print("INPUTS",inputs,label)
        buff_label = io.BytesIO()
        torch.save(label, buff_label)
        label_tensor_data = buff_label.getvalue()
        buff_label.close()
        label_tensor = worker_service_pb2.Tensor(
            data=label_tensor_data)
        label_response = self.cluster[-1].get_stub().set_label(label_tensor)
        # print(label_response)

        # print(label_response)
        # # tensor_data = inputs.numpy().tobytes()
        buff_input = io.BytesIO()
        torch.save(inputs, buff_input)
        tensor_data = buff_input.getvalue()
        buff_input.close()
        input_tensor = worker_service_pb2.Tensor(
            data=tensor_data)

        # print(input_tensor)
        self.cluster[0].get_stub().forward(
            worker_service_pb2.ForwardMessage(input=input_tensor))

    def _get_dataset(self, gen_dataset: typing.Union[typing.Callable[[], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]], None] = None):
        # Define transformations to apply to the dataset

        if gen_dataset is not None:
            self.train_loader, self.val_loader, self.test_loader = gen_dataset()
            return

        # transform = transforms.Compose([
        #     transforms.ToTensor(),  # Convert images to tensors
        #     # Normalize the pixel values to range [-1, 1]
        #     transforms.Normalize((0.5,), (0.5,))
        # ])

        # # Load the MNIST dataset
        # train_dataset = datasets.MNIST(
        #     root='./data', train=True, download=True, transform=transform)
        # test_dataset = datasets.MNIST(
        #     root='./data', train=False, download=True, transform=transform)

        # # Define the size of the training and testing sets
        # train_size = int(0.8 * len(train_dataset))
        # test_size = len(train_dataset) - train_size

        # # Split the training dataset into train and validation sets
        # train_dataset, val_dataset = torch.utils.data.random_split(
        #     train_dataset, [train_size, test_size])

        # # Create DataLoader objects for training, validation, and testing sets
        # self.train_loader = DataLoader(
        #     train_dataset, batch_size=64, shuffle=True)
        # self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        # self.test_loader = DataLoader(
        #     test_dataset, batch_size=64, shuffle=False)
        # # Print dataset sizes

        # print(f"Training set size: {len(train_dataset)}")
        # print(f"Validation set size: {len(val_dataset)}")
        # print(f"Test set size: {len(test_dataset)}")
        # """
        # Training set size: 48000
        # Validation set size: 12000
        # Test set size: 10000
        # """
def download_and_load_mnist():
    # URLs for the MNIST dataset
    
    urls = {
        'train_images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'train_labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'test_images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    # Create a directory to store the dataset
    os.makedirs('./mnist', exist_ok=True)
    
    # Download the files
    for key, url in urls.items():
        response = requests.get(url, stream=True)
        with open(f'./mnist/{key}.gz', 'wb') as f:
            f.write(response.content)
    
    # Helper function to load the dataset
    def load_mnist_images(file_path):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 1, 28, 28)
        return torch.tensor(data, dtype=torch.float32) / 255.0

    def load_mnist_labels(file_path):
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return torch.tensor(data, dtype=torch.int64)

    # Load train and test data
    train_images = load_mnist_images('./mnist/train_images.gz')
    train_labels = load_mnist_labels('./mnist/train_labels.gz')
    test_images = load_mnist_images('./mnist/test_images.gz')
    test_labels = load_mnist_labels('./mnist/test_labels.gz')
    
    return train_images, train_labels, test_images, test_labels

def get_dataset(batch_size=64, val_split=0.2):
    # Load MNIST data
    train_images, train_labels, test_images, test_labels = download_and_load_mnist()
    
    # Create TensorDatasets
    train_dataset = TensorDataset(train_images, train_labels)
    
    # Split the training data into training and validation sets
    num_train = len(train_dataset)
    num_val = int(val_split * num_train)
    num_train -= num_val
    
    train_dataset, val_dataset = random_split(train_dataset, [num_train, num_val])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def serve():
    port = sys.argv[1]
    criterion = {"name": "CrossEntropyLoss", "params": {}}

    optimizer = {"name": "Adam", "params": {
        "lr": 0.001, "weight_decay": 0.0001}}

    cluster = [Device("dev1", "localhost", "50052"),
            Device("dev2", "localhost", "50053")]

    model  = SimpleCNN()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    coordinator_service_pb2_grpc.add_CoordinatorServiceServicer_to_server(
        Dispatcher(model, criterion, optimizer, max_epochs=1,
                        gen_dataset=get_dataset, cluster=cluster, init_state_dict=True), server)
    server.add_insecure_port('[::]:'+port)
    print("Server started at port "+port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()



# dispatcher._get_dataset(SimpleCNN(), gen_dataset=a,)
# print(dispatcher.test_loader, dispatcher.train_loader, dispatcher.val_loader)

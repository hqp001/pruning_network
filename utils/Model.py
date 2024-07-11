import torch.nn as nn
import torch
import torch.nn.utils.prune as prune
from onnx2torch.node_converters.matmul import OnnxMatMul
from onnx2torch import convert
import math


# Neural Network Configuration
# Sequential is a container module in PyTorch that encapsulates a series of layers
# this setup defines a simple feed-forward nn with two hidden layers
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = [
            nn.Linear(2 * 2, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 10),
            nn.Softmax(dim=0)
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

    def get_weights(self):
        weights = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                weights[name] = param.data
        return weights

    def count_parameters(self):
        # total_params = sum(p.numel() for p in model.parameters())
        # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        non_zero_weights = 0

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                non_zero_weights += torch.count_nonzero(layer.weight).item()

        return non_zero_weights

    def initialize_weights(self, initial_weights):
        pruned_state_dict = self.state_dict()

        for parameter_name, parameter_values in initial_weights.items():
            # Pruned weights are called <parameter_name>_orig
            augmented_parameter_name = parameter_name + "_orig"

            if augmented_parameter_name in pruned_state_dict:
                pruned_state_dict[augmented_parameter_name] = parameter_values
            else:
                # Parameter name has not changed
                # e.g. bias or weights from non-pruned layer
                pruned_state_dict[parameter_name] = parameter_values

        self.load_state_dict(pruned_state_dict)

        return self

    def apply_mask(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                prune.remove(layer, 'weight')

class TestNano(SimpleNN):

    def __init__(self):

        super(TestNano, self).__init__()

        self.layers = [
            nn.Linear(10, 10, bias=False),
            nn.ReLU()
        ]

        self.model = nn.Sequential(*self.layers)

class MNIST_NET_256x(SimpleNN):
    def __init__(self, num_hidden_layers = 2):

        super(MNIST_NET_256x, self).__init__()

        self.layers = [
            nn.Linear(28 * 28, 256),
            nn.ReLU()
        ]

        for i in range(num_hidden_layers - 1):

            self.layers.extend([
                nn.Linear(256, 256),
                nn.ReLU()
            ])

        self.layers.append(nn.Linear(256, 10))

        self.model = nn.Sequential(*self.layers)

class ONNXModel(SimpleNN):

    def __init__(self, graph_module):

        super(ONNXModel, self).__init__()

        layers = []

        state_dict = graph_module.state_dict()

        attr_nodes = {}  # To store the get_attr nodes

        for node in graph_module.graph.nodes:

            if node.op == 'placeholder':
                continue

            elif node.op == 'get_attr':
                attr_nodes[node.name] = node.target  # Store the target name for get_attr nodes


            elif node.op == 'call_module':
                submod = dict(graph_module.named_modules())[node.target]
                if isinstance(submod, nn.Linear):
                    layers.append(submod)

                elif isinstance(submod, nn.ReLU):
                    layers.append(nn.ReLU())

                elif isinstance(submod, OnnxMatMul):

                    weight = state_dict[attr_nodes[node.args[1].name]]

                    weight_tensor = weight
                    out_features, in_features = weight_tensor.shape

                    linear_layer = nn.Linear(in_features, out_features, bias=False)
                    linear_layer.weight.data = weight_tensor

                    layers.append(linear_layer)

                elif isinstance(submod, nn.Flatten):

                    layers.append(submod)

                elif isinstance(submod, nn.Conv2d):

                    layers.append(submod)

                    #continue

                else:

                    raise ValueError(f"Unknown module: {submod}")

            elif node.op == 'call_function':
                if node.target == torch.relu:
                    layers.append(nn.ReLU())

                else:
                    submod = dict(graph_module.named_modules())[node.target]
                    raise ValueError(f"Unknown module: {submod}")

            elif node.op == 'output':
                break



        self.layers = layers

        self.model = nn.Sequential(*self.layers)

class TorchModel(SimpleNN):

    def __init__(self, file_name):

        super(TorchModel, self).__init__()

        layers = []

        model = torch.load(file_name)

class DoubleModel(SimpleNN):

    def __init__(self, layers):

        super(DoubleModel, self).__init__()

        new_layers = []

        for layer in layers:
            if isinstance(layer, nn.ReLU):
                new_layers.append(nn.ReLU())
            elif isinstance(layer, nn.Softmax):
                new_layers.append(nn.Softmax(dim=layer.dim))
            elif isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                if in_features == 256:
                    in_features = 512
                if out_features == 256:
                    out_features = 512
                new_linear_layer = nn.Linear(in_features, out_features)

                # Copy existing weights to the new layer
                new_linear_layer.weight.data[:layer.out_features, :layer.in_features] = layer.weight.data
                new_linear_layer.bias.data[:layer.out_features] = layer.bias.data

                print(layer.weight)
                print(new_linear_layer.weight)
                print(layer.weight.size())
                print(new_linear_layer.weight.size())
                # Randomly initialize the rest of the weights and biases
                nn.init.kaiming_uniform_(new_linear_layer.weight[layer.out_features:, layer.in_features:], a=math.sqrt(5))
                nn.init.uniform_(new_linear_layer.bias[layer.out_features:])

                new_layers.append(new_linear_layer)
            else:
                raise ValueError("Unknown layer")

        self.layers = new_layers
        self.model = nn.Sequential(*self.layers)

#import onnx

#model = convert(onnx.load("./vnncomp2022_benchmarks/benchmarks/cifar100_tinyimagenet_resnet/onnx/CIFAR100_resnet_small.onnx"))
#print(model)


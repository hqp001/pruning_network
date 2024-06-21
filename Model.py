import torch.nn as nn
import torch
import torch.nn.utils.prune as prune


# Neural Network Configuration
# Sequential is a container module in PyTorch that encapsulates a series of layers
# this setup defines a simple feed-forward nn with two hidden layers

HIDDEN_LAYERS_SIZE=500

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layers = [
            nn.Linear(28 * 28, HIDDEN_LAYERS_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS_SIZE, 10),
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

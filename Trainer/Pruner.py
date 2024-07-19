import torch.nn.utils.prune as prune
from torch import nn
import torch
import torch.nn.functional as functional
import tqdm
import numpy as np

from utils.Model import count_params

class Pruner:

    def __init__(self, model, sparsity = 0):

        self.sparsity = sparsity

        self.model = model

    def prune(self):

        total_parameters = count_params(self.model)

        parameters_to_prune = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.sparsity)

        print("Pruned: ", total_parameters - count_params(self.model))

        return self


    def get_model(self):

        return self.model

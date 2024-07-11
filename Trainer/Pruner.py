from skorch import NeuralNetClassifier
import torch.nn.utils.prune as prune
from torch import nn
import torch
import torch.nn.functional as functional
import tqdm
import numpy as np


class Pruner:

    def __init__(self, model, sparsity = 0):

        self.sparsity = sparsity

        self.model = model

    def prune(self):

        total_parameters = self.model.count_parameters()

        parameters_to_prune = [(module, 'weight') for module in self.model.model if isinstance(module, nn.Linear)]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.sparsity)

        print("Pruned: ", total_parameters - self.model.count_parameters())

        return self


    def get_model(self):

        return self.model

from skorch import NeuralNetClassifier
import torch.nn.utils.prune as prune
from torch import nn
import torch
import torch.nn.functional as functional
import tqdm
import numpy as np


# Train the original model
class ModelTrainer:
    def __init__(self, model, max_epochs, learning_rate, device):
        self.model = model
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.device = device

    def train(self, data_loader, l1_reg=0.01, l2_reg=0):

        #optimiser = torch.optim.Adam(params=self.model.parameters(), weight_decay=1e-4, lr=1e-3)
        optimiser = torch.optim.SGD(params=self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.max_epochs):

            print(f"Training epoch: {epoch + 1}/{self.max_epochs}")
            iter_train = iter(data_loader)

            num_correct = 0
            num_samples = 0

            pbar = tqdm.tqdm(range(len(data_loader)))
            pbar.set_description(f"Training Accuracy: {0}")

            for _ in pbar:

                x, y = next(iter_train)
                x = x.reshape((x.shape[0], -1))

                self.model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # print(self.model(x)[0])

                #scores = self.model(x)
                scores = functional.log_softmax(self.model(x), dim=1)

                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                train_acc = num_correct / num_samples
                pbar.set_description(f"Epoch Accuracy: {train_acc:.2f}")

                l1_regularization_loss = 0
                l2_regularization_loss = 0
                for param in self.model.parameters():
                    l1_regularization_loss += torch.sum(torch.abs(param))
                    l2_regularization_loss += torch.sum(param ** 2)

                loss = (functional.cross_entropy(scores, y) +
                        l1_reg * l1_regularization_loss +
                        l2_reg * l2_regularization_loss)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # val_acc = self.check_accuracy(dataset=1)[2]

            # if epoch % 5 == 0:
            #     print(f"Validation accuracy: {val_acc:.4f}")

        return (num_correct / num_samples).item()

    def retrain(self, n_epochs, learning_rate, data_loader, l1_reg=0.01, l2_reg=0):

        optimiser = torch.optim.SGD(params=self.model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):

            print(f"Training epoch: {epoch + 1}/{self.max_epochs}")
            iter_train = iter(data_loader)

            num_correct = 0
            num_samples = 0

            pbar = tqdm.tqdm(range(len(data_loader)))
            pbar.set_description(f"Training Accuracy: {0}")

            for _ in pbar:

                x, y = next(iter_train)
                x = x.reshape((x.shape[0], -1))

                self.model.train()
                x = x.to(device=self.device)
                y = y.to(device=self.device)

                # print(self.model(x)[0])

                scores = self.model(x)
                #scores = functional.log_softmax(self.model(x), dim=1)

                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
                train_acc = num_correct / num_samples
                pbar.set_description(f"Epoch Accuracy: {train_acc:.2f}")

                l1_regularization_loss = 0
                l2_regularization_loss = 0
                for param in self.model.parameters():
                    l1_regularization_loss += torch.sum(torch.abs(param))
                    l2_regularization_loss += torch.sum(param ** 2)

                loss = (functional.cross_entropy(scores, y) +
                        l1_reg * l1_regularization_loss +
                        l2_reg * l2_regularization_loss)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # val_acc = self.check_accuracy(dataset=1)[2]

            # if epoch % 5 == 0:
            #     print(f"Validation accuracy: {val_acc:.4f}")

        return (num_correct / num_samples).item()

    def count_neurons(self, data_loader, digit=8):

        iter_train = iter(data_loader)

        num_correct = 0
        num_samples = 0

        pbar = tqdm.tqdm(range(len(data_loader)))
        pbar.set_description(f"Training Accuracy: {0}")

        total_neurons_freq = None
        total_weights_freq = None
        total_inputs = 0
        abs_weights = []

        for layer in self.model.layers:
            if isinstance(layer, torch.nn.Linear):
                abs_weights.append(np.abs(layer.weight.data.numpy()))


        for _ in pbar:


            activations = {}
            def hook_fn(module, input, output):
                #print(module)
                #print(input.shape)
                activations[module] = output.detach().numpy()


            # Register hooks to all layers you want to inspect
            for layer in self.model.model.children():
                layer.register_forward_hook(hook_fn)

            iter_train = iter(data_loader)


            x, y = next(iter_train)

            x = x.reshape((x.shape[0], -1))

            if digit < 10:

                mask = (y == digit)
                x = x[mask]
                y = y[mask]

            total_inputs += len(x)

            output = self.model(x)

            neuron_freq = []
            weight_freq = []

            last_activation = [np.ones((len(x), 784), dtype=int)]

            for layer, activation in activations.items():
                output_activation = activation
                output_activation = np.where(output_activation > 0, 1, 0).astype(int)

                if isinstance(layer, torch.nn.ReLU):
                    neuron_freq.append(np.sum(output_activation, axis=0))
                    last_activation.append(output_activation)

                if isinstance(layer, torch.nn.Linear):
                    weight_matrix = layer.weight.data.numpy()
                    abs_weight_matrix = np.ones_like(weight_matrix, dtype=int)

                    weight_matrix_sum = np.zeros_like(weight_matrix, dtype=int)

                    for i in range(output_activation.shape[0]):

                        to_activated = output_activation[i]

                        from_activated = last_activation[-1][i]

                        #print(to_activated.shape)
                        #print(from_activated.shape)

                        activated_weight_matrix = abs_weight_matrix * np.outer(to_activated, from_activated)

                        weight_matrix_sum += activated_weight_matrix

                    weight_freq.append(weight_matrix_sum)

            if total_neurons_freq == None:
                total_neurons_freq = neuron_freq
            else:
                for idx, layer in enumerate(total_neurons_freq):
                    layer += neuron_freq[idx]

            if total_weights_freq == None:
                total_weights_freq = weight_freq
            else:
                for idx, layer in enumerate(total_weights_freq):
                    layer += weight_freq[idx]

            #print(neuron_freq)
            #print(weight_freq)

        for i in range(len(total_neurons_freq)):
            total_neurons_freq[i] = total_neurons_freq[i] / total_inputs
            #print(total_neurons_freq)
        for i in range(len(total_weights_freq)):
            total_weights_freq[i] = total_weights_freq[i] / total_inputs

        #print(len(total_neurons_freq))
        #print(len(total_weights_freq))
        #print(total_neurons_freq[0].shape)
        #print(total_weights_freq[0].shape)

        print("Total_inputs: ", total_inputs)

        return self.model, total_neurons_freq, total_weights_freq, abs_weights

    def calculate_score(self, data_loader):

        num_correct = 0
        num_samples = 0

        self.model.eval()

        with torch.no_grad():
            for x, y in data_loader:

                x = x.to(device=self.device)
                y = y.to(device=self.device)
                x = x.reshape((x.shape[0], -1))

                # noinspection PyCallingNonCallable
                scores = self.model(x)
                _, predictions = scores.max(1)
                #print(predictions)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

            acc = float(num_correct) / num_samples

        return acc

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

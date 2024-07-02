from skorch import NeuralNetClassifier
import torch.nn.utils.prune as prune
from torch import nn
import torch
import torch.nn.functional as functional
import tqdm



# Train the original model
class ModelTrainer:
    def __init__(self, model, max_epochs, learning_rate, device):
        self.model = model
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.device = device

    def train(self, data_loader, l1_reg=0, l2_reg=0):

        optimiser = torch.optim.Adam(params=self.model.parameters(), weight_decay=1e-4, lr=1e-3)

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

    def retrain(self, data_loader, l1_reg=0, l2_reg=0):

        optimiser = torch.optim.SGD(params=self.model.parameters(), lr=1e-2)

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

from skorch import NeuralNetClassifier
import torch.nn.utils.prune as prune
from torch import nn
import torch


# Train the original model
class ModelTrainer:
    def __init__(self, model, max_epochs, learning_rate, device):
        self.model = model
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.clf = NeuralNetClassifier(
            self.model, 
            max_epochs=self.max_epochs, 
            lr=self.learning_rate,
            optimizer=torch.optim.SGD,
            criterion=nn.CrossEntropyLoss,
            batch_size=64,
            iterator_train__shuffle=True,
            device=device
        )
    
    def train(self, X, y):

        self.model.train()

        self.clf.fit(X, y)

        print("Training Score:")
        training_score = self.calculate_score(X, y)
        print(training_score)

        return training_score

    def calculate_score(self, x, y):

        self.model.eval()

        return self.clf.score(x, y)

class Pruner:

    def __init__(self, model, trainer, sparsity = 0):

        self.sparsity = sparsity

        self.model = model

        self.trainer = trainer

    def prune(self):

        total_parameters = self.model.count_parameters()

        parameters_to_prune = [(module, 'weight') for module in self.model.model if isinstance(module, nn.Linear)]
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.sparsity)

        print("Pruned: ", total_parameters - self.model.count_parameters())

        return self

    def fine_tune(self, X_train, y_train):

        # print(self.model.get_weights())

        self.trainer.train(X=X_train, y=y_train)

        # print(self.model.get_weights())

        for layer in self.model.layers:
            if isinstance(layer, nn.Linear):
                prune.remove(layer, 'weight')



        return self

    
    def get_model(self):

        return self.model

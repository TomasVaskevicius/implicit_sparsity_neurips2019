import numpy as np
import torch

from .trainer import Trainer
from ..config.config import PyTorchConfig

class SgdHyperparameters:
    """ A class for defining hyperparameters to be used by SgdTrainer. """

    def __init__(self, learning_rate, batch_size):
        self.lr = learning_rate
        self.batch_size = batch_size

class SgdTrainer(Trainer):
    """ A class for training a model using SGD on a given dataset. """

    def __init__(
            self,
            n,
            dataset_generator,
            model,
            hyperparameters,
            pytorch_config = PyTorchConfig()):
        """
        n is the dataset size.
        dataset_generator should be a subclass of core.data.dataset.
        model should be a subclass of core.model.model.
        hyperparameters should be of type SgdHyperparameters.
        PyTorchConfig specified the datatype and device for pytorch tensors.
        """
        super().__init__(n, dataset_generator, model, pytorch_config)
        self.batch_size = hyperparameters.batch_size
        self.lr = hyperparameters.lr


    def do_one_epoch(self):
        """ Performs one sgd epoch with the given batch size and notifies
        observers after every iteration. """
        n = self.X.size()[0]
        shuffle_idx = torch.randperm(n)

        for i in range(0, n, self.batch_size):
            self.notify_all_observers()
            self.model.train()

            X_batch = self.X[shuffle_idx[i : i + self.batch_size]]
            y_batch = self.y[shuffle_idx[i : i + self.batch_size]]
            y_pred = self.model(X_batch)

            optimizer = torch.optim.SGD(
                    self.model.parameters(), lr = self.lr)
            loss_criterion = self.model.get_loss_criterion()
            loss = loss_criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



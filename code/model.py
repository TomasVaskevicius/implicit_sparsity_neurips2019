import core
import torch


class ImplicitLassoLinearRegression(core.Model):
    """ Instead of representing the hidden layer as a vector w, we will
    represent it as w = (u_{1}^{2} - v_{1}^{2}, ..., u_{d}^{2} - v_{d}^{2})
    and we will do gradient descent on u and v.
    """

    def __init__(self, d):
        """ Parameter d specifies the number of parameters. By default, bias
        parameter is not allowed, and instead the covariates matrix could be
        augmented by a column of ones."""
        super().__init__()
        self.layer = torch.nn.Linear(d, 1, bias=False)
        self.layer2 = torch.nn.Linear(d, 1, bias=False)

    def forward(self, x):
        squared_weight = self.layer.weight**2 - self.layer2.weight**2
        y = x.matmul(squared_weight.t())
        return y

    def get_loss_criterion(self):
        return torch.nn.MSELoss()

    def get_parameters(self):
        return self.layer.weight**2 - self.layer2.weight**2

    def init_weights(self, alpha):
        torch.nn.init.ones_(self.layer.weight)
        torch.nn.init.ones_(self.layer2.weight)
        self.layer.weight.data *= alpha
        self.layer2.weight.data *= alpha

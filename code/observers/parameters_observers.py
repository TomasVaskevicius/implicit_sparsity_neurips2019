import core


class ParametersObserver(core.Observer):
    """ A class for observing the true parameters w. """

    def __init__(self, frequency=1):
        super().__init__(frequency)
        self.parameters = []

    def _notify(self, trainer):
        p = trainer.model.get_parameters().cpu().detach().numpy()
        self.parameters.append(p)

    @staticmethod
    def aggregate_results(parameters_observer):
        params = core._aggregate_numeric_results(
            parameters_observer, 'parameters')
        return {
            'params': params.squeeze()
        }


class UVObserver(core.Observer):
    """ A class for observing u and v parameters during the training. """

    def __init__(self, frequency=1):
        super().__init__(frequency)
        self.u = []
        self.v = []

    def _notify(self, trainer):
        u = trainer.model.layer.weight.data.cpu().detach().numpy()
        v = trainer.model.layer2.weight.data.cpu().detach().numpy()
        self.u.append(u)
        self.v.append(v)

    @staticmethod
    def aggregate_results(uv_observers):
        us = core._aggregate_numeric_results(uv_observers, 'u')
        vs = core._aggregate_numeric_results(uv_observers, 'v')
        return {
            'us': us.squeeze(),
            'vs': vs.squeeze()
        }

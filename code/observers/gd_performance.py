from observers.performance_base import *


class TruePerformanceObserver(PerformanceObserverBase):
    """ A class for observing the true parameters w. """

    def __init__(self, frequency=1):
        super().__init__(frequency)

    def _notify(self, trainer):
        w_t = trainer.model.get_parameters().cpu().detach().numpy()
        self.append_performance_metrics(w_t, trainer)

    @staticmethod
    def aggregate_results(true_performance_observers):
        return PerformanceObserverBase \
            .aggregate_performance_results(true_performance_observers)

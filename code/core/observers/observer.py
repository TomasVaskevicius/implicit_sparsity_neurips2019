from abc import ABC, abstractmethod

class Observer(ABC):
    """ An abstract base class for all observers of child classes of
    core.observer.observable. The notification policy depends on
    the particular observable. """

    def __init__(self, frequency):
        """ The frequency parameters controls how often we want notify function
        to do actual work. For example, if frequency == 10, then all but every
        10th call to notify will be ignored. """
        super().__init__()
        self.__frequency = frequency
        self.__iterations = 0
        self._is_set_up = False

    def notify(self, observable):
        """ A wrapper against the true notify method. The purpose of this
        method is to filter out some of the calls to notify. """
        if not self._is_set_up:
            self._set_up(observable)
            self._is_set_up = True

        if self.__iterations % self.__frequency == 0:
            self._notify(observable)
        self.__iterations = (self.__iterations + 1) % self.__frequency

    def _set_up(self, observable):
        """ This method will be called only once, before the _notify method
        is ever called. """
        pass

    def clean_up(self, observable):
        """ This method will be called after training allowing free some of
        the resources. For example, there are some problems with
        multiprocessing and storing pytorch tensors as attributes. Hence
        any attributes contining torch tensors should be cleaned up here
        or converted to numpy. """
        pass

    @abstractmethod
    def _notify(self, observable):
        """ The actual implementation of notify method. """
        pass

    @staticmethod
    def aggregate_results(observers):
        """ Given a list of observers of the same type, aggregates and returns
        their results. """
        return None


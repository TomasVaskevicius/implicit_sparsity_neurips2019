from abc import ABC, abstractmethod

class Observable(ABC):
    """ An abstract base class for all runners.
    For not it only enforces all runners to support observers. """

    def __init__(self):
        super().__init__()
        self.observers = []

    def register_observer(self, observer):
        """ Appends the given observer to self.observers array. """
        self.observers.append(observer)

    def notify_all_observers(self):
        """ Notifies all the observers with the current state of the trainer
        which should include all the relevand information (i.e. state of the
        model, dataset_generator, training loss, etc). """
        for observer in self.observers:
            observer.notify(self)

    def clean_up_all_observers(self):
        """ Notifies all observers to clean up. After this method is called,
        no observer sould be notified anymore. """
        for observer in self.observers:
            observer.clean_up(self)

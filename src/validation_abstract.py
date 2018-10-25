import pickle
from abc import ABC, abstractmethod


class ValidationAbstract(ABC):

    def __init__(self, cache_path):
        self.actual = None
        self.predicted = None
        self.available_classes = None
        self.cache_path = cache_path
        self._load(cache_path)

    def save(self):
        if self.cache_path is not None:
            try:
                pickle.dump((self.actual, self.predicted, self.available_classes), open(self.cache_path, "wb"))
            except Exception as e:
                print(e)
                print("Unable to store analyzer contents...")
        else:
            print("No save path available for validation data...")
        return

    def _load(self, cache_path):
        if cache_path is not None:
            try:
                self.actual, self.predicted, self.available_classes = pickle.load(open(cache_path, "rb"))
            except FileNotFoundError:
                print("Unable to load analyzer contents...")
            except Exception as e:
                print(e)
                print("Error loading validation data...")
        return

    @abstractmethod
    def cross_validation_summary(self, folder_path, model_class):
        raise NotImplementedError

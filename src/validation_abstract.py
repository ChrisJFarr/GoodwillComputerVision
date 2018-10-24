from abc import ABC, abstractmethod


class ValidationAbstract(ABC):

    @abstractmethod
    def cross_validation_summary(self, folder_path, model_class):
        raise NotImplementedError

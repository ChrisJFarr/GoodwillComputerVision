from abc import ABC, abstractmethod


class ImageClassificationAbstract(ABC):

    def __init__(self):
        self._model = None

    # Methods:
    @staticmethod
    def get_images_array(image_paths_list):
        # accepts image paths list, returns images array
        raise NotImplementedError

    @staticmethod
    def get_classes_array(image_paths_list):
        # accepts image path, returns image classes
        raise NotImplementedError

    def set_model(self, model):
        self._model = model

    def get_model(self):
        return self._model

    # Abstract Methods:
    @abstractmethod
    def train(self, image_paths_list):
        # accepts list of image paths, trains model, stores trained model
        return None

    @staticmethod
    @abstractmethod
    def preprocess_images(images_array):
        # accepts images array, return preprocessed images array
        return images_array

    @abstractmethod
    def predict(self, image_paths_list):
        # accepts list of image paths, returns predicted classes
        return image_paths_list

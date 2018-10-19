from abc import ABC, abstractmethod
import cv2
import numpy as np


class ImageClassificationAbstract(ABC):

    def __init__(self):
        self._model = None

    # Methods:
    @staticmethod
    def get_images_array(image_paths_list):
        # accepts image paths list, returns images array
        image_list = []
        for file in image_paths_list:
            image = cv2.imread(file)
            assert image is not None, "Issue loading file:" + file
            image_list.append(image)
        return np.array(image_list)

    def set_model(self, model):
        self._model = model

    def get_model(self):
        assert self._model is not None, "Accessing model prior to training."
        return self._model

    # Abstract Methods:
    @staticmethod
    @abstractmethod
    def get_classes_array(image_paths_list):
        # accepts image path, returns image classes
        return image_paths_list

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

import unittest
from src.models.model_abstract import ImageClassificationAbstract


class ModelAbstractTestShell(ImageClassificationAbstract):
    # Abstract Method Shells:
    def train(self, image_paths_list):
        # accepts list of image paths, trains model, stores trained model
        return None

    @staticmethod
    def preprocess_images(images_array):
        # accepts images array, return preprocessed images array
        return images_array

    def predict(self, image_paths_list):
        # accepts list of image paths, returns predicted classes
        return image_paths_list


class TestImageClassificationAbstract(unittest.TestCase):

    def test_get_images_array(self):
        # accepts image paths list, returns images array
        raise NotImplementedError

    def test_get_classes_array(self):
        # accepts image path, returns image classes
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()

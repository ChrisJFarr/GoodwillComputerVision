from src.models.size_classification_model import SizeClassificationModel
import unittest


class TestSizeClassificationModel(unittest.TestCase):
    def setUp(self):
        pass

    # Override Abstract Methods:
    def test_train(self):
        # accepts list of image paths, trains model, stores trained model
        raise NotImplementedError

    def test_preprocess_images(self):
        # accepts images array, return preprocessed images array
        raise NotImplementedError

    def test_predict(self):
        # accepts list of image paths, returns predicted classes
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()

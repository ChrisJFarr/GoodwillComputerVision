from src.models.type_classification_model import TypeClassificationModel
import unittest


class TestTypeClassificationModel(unittest.TestCase):
    def setUp(self):
        pass

    # Override Abstract Methods:
    def test_get_classes_array(self):
        # accepts image path, returns image classes
        raise NotImplementedError

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

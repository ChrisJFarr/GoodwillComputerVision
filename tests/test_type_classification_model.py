from src.models.type_classification_model import TypeClassificationModel
import unittest
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

TEST_SIZE_FILENAMES = ['mensjeans_3xl_good_avenue_11.JPG',
                       'sweats_large_good_adidas_13.jpg',
                       'womensdresses_large_poor_danny_nicole_2.jpg',
                       'womensshorts_large_good_worthington_27.jpg',
                       'womenssleevelessshirt_large_good_elle_7.JPG'
                       ]
ACTUAL_TEST_CLASSES = ["mensjeans", "sweats", "womensdresses", "womensshorts", "womenssleevelessshirt"]
TYPE_CLASSIFICATION_DATA = Path("src/data/type_data/train")


class TestTypeClassificationModel(unittest.TestCase):

    type_classification_model = TypeClassificationModel()

    def setUp(self):
        pass

    # Override Abstract Methods:

    def test_get_classes_array(self):
        # accepts image path, returns image classes
        test_result = self.type_classification_model.get_classes_array(TEST_SIZE_FILENAMES)
        y_data = ACTUAL_TEST_CLASSES
        self.assertTrue(all([a == b for a, b in zip(y_data, test_result)]),
                        "Error parsing type classes from file names")

    def test_train(self):
        # accepts list of image paths, trains model, stores trained model
        # Read file names, Create image paths
        image_paths = []
        for root, dirs, files in os.walk(TYPE_CLASSIFICATION_DATA):
            for name in files:
                image_paths.append(os.path.join(root, name))
        np.random.shuffle(image_paths)
        # Test train model
        self.type_classification_model.train(image_paths[0:50])
        # Assert model is saved (get_model)
        model = self.type_classification_model.get_model()
        self.assertTrue(isinstance(model, LinearSVC),
                        "Unexpected type. Expected %s received %s" % (str(type(LinearSVC)), str(type(model))))

    def test_predict(self):
        # accepts list of image paths, returns predicted classes
        # Use real data TODO convert to test dataset when available
        image_paths = []
        for root, dirs, files in os.walk(TYPE_CLASSIFICATION_DATA):
            for name in files:
                image_paths.append(os.path.join(root, name))
        np.random.seed(100)
        np.random.shuffle(image_paths)
        # Train model
        self.type_classification_model.train(image_paths[0:100])
        # Generate predictions
        predictions = self.type_classification_model.predict(image_paths[100:120])
        # Extract labels
        file_names = [os.path.basename(image_path) for image_path in image_paths[100:120]]
        y_data = self.type_classification_model.get_classes_array(file_names)
        # Assert baseline accuracy
        print()
        print("Type classification train accuracy:", accuracy_score(y_data, predictions))
        print()
        self.assertTrue(accuracy_score(y_data, predictions) > .6,
                        "Diminishing train score.")

    # def test_preprocess_images(self):
    #     # accepts images array, return preprocessed images array
    #     raise NotImplementedError
    #
    # def test_hog_feature_extractor(self):
    #     raise NotImplementedError
    #
    # def test_lbp_feature_feature_extractor(self):
    #     raise NotImplementedError

    def test_create_augmented_data(self):
        image_paths = []
        for root, dirs, files in os.walk(TYPE_CLASSIFICATION_DATA):
            for name in files:
                image_paths.append(os.path.join(root, name))
        np.random.seed(100)
        np.random.shuffle(image_paths)
        file_names = [os.path.basename(image_path) for image_path in image_paths[:20]]
        image_paths = image_paths[:20]
        x_data = self.type_classification_model.get_images_array(image_paths)
        # Preprocess x_data
        x_data = self.type_classification_model.preprocess_images(x_data)
        # Extract y_data
        y_data = self.type_classification_model.get_classes_array(file_names)
        # Augment data
        x_data, y_data = self.type_classification_model.created_augmented_data(x_data, y_data)
        # Assert data types
        self.assertTrue(isinstance(x_data, np.ndarray),
                        "x_data type expected %s received %s" % (type(np.ndarray), type(x_data)))
        self.assertTrue(isinstance(y_data, np.ndarray),
                        "y_data type expected %s received %s" % (type(np.ndarray), type(y_data)))
        # TODO Assert y_data element string instance


if __name__ == "__main__":
    unittest.main()

from src.models.size_classification_model import SizeClassificationModel, TARGET_SIZE, SIZE_MAP
import unittest
import os
from random import shuffle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# rando_list = os.listdir(os.path.join(os.getcwd(), "src", "data", "size_data", "train", "t_shirt"))
# shuffle(rando_list)
# rando_list

# Test Support Constants
TEST_SIZE_FILENAMES = ['tshirts_medium_good_anntaylorloft_82.jpg', 'tshirts_small_poor_fruitoftheloom_108.jpg',
                       'tshirts_medium_poor_mossimo_84.jpg', 'tshirts_2xl_good_fadedglory_71.jpg',
                       'tshirts_large_poor_champion_11.jpg']
ACTUAL_TEST_SIZES = ["medium", "small", "medium", "2xl", "large"]
SIZE_CLASSIFICATION_DATA = "src\\data\\size_data\\train\\t_shirt"
# SIZE_CLASSIFICATION_DATA = "..\\src\\data\\size_data\\train\\t_shirt"


class TestSizeClassificationModel(unittest.TestCase):

    size_classification_model = SizeClassificationModel()

    def setUp(self):
        pass

    # Override Abstract Methods:
    def test_get_classes_array(self):
        # accepts image path, returns image classes
        test_result = self.size_classification_model.get_classes_array(TEST_SIZE_FILENAMES)
        y_data = [SIZE_MAP.get(sz.lower()) for sz in ACTUAL_TEST_SIZES]
        self.assertTrue(all([a == b for a, b in zip(y_data, test_result)]),
                        "Error parsing size classes from file names")

    def test_train(self):
        # accepts list of image paths, trains model, stores trained model
        # Read file names
        test_file_names = os.listdir(SIZE_CLASSIFICATION_DATA)
        # Create image paths
        image_paths = [os.path.join(SIZE_CLASSIFICATION_DATA, file_name) for file_name in test_file_names]
        # Test train model
        self.size_classification_model.train(image_paths)
        # Assert model is saved (get_model)
        model = self.size_classification_model.get_model()
        self.assertTrue(isinstance(model, LinearSVC),
                        "Unexpected type. Expected %s received %s" % (str(type(LinearSVC)), str(type(model))))

    def test_preprocess_images(self):
        # accepts images array, return preprocessed images array
        # Use real data
        test_file_names = os.listdir(SIZE_CLASSIFICATION_DATA)
        image_paths = [os.path.join(SIZE_CLASSIFICATION_DATA, file_name) for file_name in test_file_names]
        # Load images
        images_array = SizeClassificationModel.get_images_array(image_paths)
        # Test preprocess_images
        test_array = self.size_classification_model.preprocess_images(images_array)
        # Assert TARGET_SIZE
        actual_shape = test_array.shape[1:3]
        expected_shape = (TARGET_SIZE[1], TARGET_SIZE[0])  # cv2 reverses, have to flip
        self.assertTrue(actual_shape == expected_shape,
                        "Unexpected shape returned. Received %s expected %s" % (str(actual_shape), str(expected_shape)))

    def test_predict(self):
        # accepts list of image paths, returns predicted classes
        # Use real data TODO convert to test dataset when available
        test_file_names = os.listdir(SIZE_CLASSIFICATION_DATA)
        image_paths = [os.path.join(SIZE_CLASSIFICATION_DATA, file_name) for file_name in test_file_names]
        # Extract labels
        y_data = self.size_classification_model.get_classes_array(test_file_names)
        # Train model
        self.size_classification_model.train(image_paths)
        # Generate predictions
        predictions = self.size_classification_model.predict(image_paths)
        # Assert baseline accuracy
        print()
        print("Size classification train accuracy:", accuracy_score(y_data, predictions))
        self.assertTrue(accuracy_score(y_data, predictions) > .6,
                        "Diminishing train score.")
        print()

    def test_create_augmented_data(self):
        test_file_names = os.listdir(SIZE_CLASSIFICATION_DATA)
        image_paths = [os.path.join(SIZE_CLASSIFICATION_DATA, file_name) for file_name in test_file_names]
        x_data = self.size_classification_model.get_images_array(image_paths)
        # Preprocess x_data
        x_data = self.size_classification_model.preprocess_images(x_data)
        # Extract y_data
        y_data = self.size_classification_model.get_classes_array(test_file_names)
        # Augment data
        x_data, y_data = self.size_classification_model.created_augmented_data(x_data, y_data)
        # Assert data types
        self.assertTrue(isinstance(x_data, np.ndarray),
                        "x_data type expected %s received %s" % (type(np.ndarray), type(x_data)))
        self.assertTrue(isinstance(y_data, np.ndarray),
                        "y_data type expected %s received %s" % (type(np.ndarray), type(y_data)))
        # TODO Assert y_data element string instance


if __name__ == "__main__":
    unittest.main()

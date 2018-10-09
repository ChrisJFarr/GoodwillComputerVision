from src.models.size_classification_model import SizeClassificationModel
import unittest
import os
from random import shuffle

# rando_list = os.listdir(os.path.join(os.getcwd(), "src", "data", "size_data", "train", "t_shirt"))
# shuffle(rando_list)
# rando_list

# Test Support Constants
TEST_SIZE_FILENAMES = ['tshirts_medium_good_anntaylorloft_82.jpg', 'tshirts_small_poor_fruitoftheloom_108.jpg',
                       'tshirts_medium_poor_mossimo_84.jpg', 'tshirts_2xl_good_fadedglory_71.jpg',
                       'tshirts_large_poor_champion_11.jpg']
ACTUAL_TEST_SIZES = ["medium", "small", "medium", "2xl", "large"]


class TestSizeClassificationModel(unittest.TestCase):

    size_classification_model = SizeClassificationModel()

    def setUp(self):
        pass

    # Override Abstract Methods:
    def test_get_classes_array(self):
        # accepts image path, returns image classes
        test_result = self.size_classification_model.get_classes_array(TEST_SIZE_FILENAMES)
        self.assertTrue(set(ACTUAL_TEST_SIZES) == test_result,
                        "Error parsing size classes from file names")

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

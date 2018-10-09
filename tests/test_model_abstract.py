import unittest
from src.models.model_abstract import ImageClassificationAbstract
import numpy as np
import shutil
import os
from PIL import Image


class ModelAbstractTestShell(ImageClassificationAbstract):
    # Abstract Method Shells:
    @staticmethod
    def get_classes_array(image_paths_list):
        # accepts image path, returns image classes
        return image_paths_list

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

    model_abstract = ModelAbstractTestShell()

    def test_get_images_array(self):
        TEST_IMAGE_COUNT = 10
        TEST_IMAGE_SHAPE = (224, 224)
        TEST_FOLDER_NAME = "test_images"
        EXPECTED_SHAPE = (TEST_IMAGE_COUNT, TEST_IMAGE_SHAPE[0], TEST_IMAGE_SHAPE[1], 3)
        # Create fake images (numpy 0's)
        image_list = [Image.new('RGB', TEST_IMAGE_SHAPE, color='black') for _ in range(TEST_IMAGE_COUNT)]
        # Create temporary folder
        folder_path = os.path.join(os.getcwd(), TEST_FOLDER_NAME)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        else:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
        # Save as pngs in a temporary folder
        image_paths_list = []
        for i, image in enumerate(image_list):
            image_path = os.path.join(folder_path, "image_%s.png" % i)
            image.save(image_path, "PNG")
            image_paths_list.append(image_path)
        # Test function: # accepts image paths list, returns images array
        # Read in images
        test_array = self.model_abstract.get_images_array(image_paths_list)
        # Assert numpy array dtype
        self.assertTrue(isinstance(test_array, np.ndarray),
                        "Unexpect dtype. Received %s expected %s" % (type(test_array), type(np.ndarray)))
        # Assert numpy expected numpy size
        self.assertTrue(test_array.shape == EXPECTED_SHAPE,
                        "Unexpected shape. Received %s expected %s" % (EXPECTED_SHAPE, test_array.shape))
        # Delete images and folder created
        shutil.rmtree(folder_path)
        return None


if __name__ == "__main__":
    unittest.main()

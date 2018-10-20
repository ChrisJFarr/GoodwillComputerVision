"""
 Command line script for demo 1 with Goodwill
 Clothing Image Classification Demonstration
 @author Chris Farr 10/4/18
"""

import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from src.models.size_classification_model import SizeClassificationModel
from src.models.type_classification_model import TypeClassificationModel

# CONSTANTS
SIZE_CLASSIFICATION_FOLDER = "t_shirt"  # womens_jeans, womens_short_sleeve, womens_long_sleeve
SIZE_DATA = "src/data/size_data"
TYPE_DATA = "src/data/type_data"
SIZE_MODEL_PATH = "src/models/cached/size.joblib"
TYPE_MODEL_PATH = "src/models/cached/type.joblib"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"

# IMPLEMENTATION
# TODO Implementation Steps:
# 1. Build shells for all of the below classes (consider unit test shells)
# 2. Pull code from FashionClassification repo for model implementation
# 3. Test and build all classes/methods/abstract methods

# TODO save and load model options for train and demo functions respectively

""" DEMO 1 IMPLEMENTATION """


class DemoClass:
    # Methods:
    def run_demo(self, model_object, train_file_paths, test_file_paths):
        if model_object.get_model() is None:
            # train model
            model_object.train(train_file_paths)
            if model_object.model_path is not None:
                print("Saving model at: %s" % model_object.model_path)
                model_object.save_model()
        # create test predictions
        predicted_classes = model_object.predict(test_file_paths)
        # get actual classes
        test_file_names = [os.path.basename(image_path) for image_path in test_file_paths]
        actual_classes = model_object.get_classes_array(test_file_names)
        # Create list of original test images
        orig = model_object.get_images_array(test_file_paths)
        preprocessed = model_object.preprocess_images(orig)
        # print model validation summary
        self.display_images(orig, preprocessed, actual_classes, predicted_classes)
        return

    @staticmethod
    def run_analyzer(model_object, train_folder_path):
        # # Perform stratified k-fold prediction
        # # And  print summary of results
        # cross_validation(train_folder_path)
        raise NotImplementedError

    @staticmethod
    def display_images(orig, preprocessed, actual_classes, predicted_classes):
        # loop through arrays
        # display original image on left, preprocessed image right, overlay label on each image
        # pressing enter progresses user one image set at a time, esc exits/completes demo
        # assert orig.shape == preprocessed.shape, "Mismatching input shapes"
        print("Click on the images to continue loop...", "")
        for i in range(orig.shape[0]):
            f = plt.figure()
            f.add_subplot(1, 2, 1)
            plt.imshow(orig[i])
            plt.title("Actual: %s" % actual_classes[i])
            f.add_subplot(1, 2, 2)
            plt.imshow(preprocessed[i], cmap="gray")
            plt.title("Predicted: %s" % predicted_classes[i])
            # plt.pause(0.05)
            plt.draw()
            plt.waitforbuttonpress()
        test_score = accuracy_score(actual_classes, predicted_classes) * 100
        output_message = "Sample test accuracy: %.1f" % test_score + "%"
        print(output_message, "")
        input("End of test images, press enter to exit...")
        return

    def run_commands(self, model_object, args, train_folder_path, test_folder_path):
        if args.run == "demo":
            self.run_demo(model_object, train_folder_path, test_folder_path)
        elif args.run == "analyzer":
            self.run_analyzer(model_object, train_folder_path)


""" DEMO 1 DRIVER """


def get_type_model_files(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for name in files:
            image_paths.append(os.path.join(root, name))
    np.random.seed(2017)
    np.random.shuffle(image_paths)
    return image_paths


def get_size_model_files(folder_path):
    file_names = os.listdir(folder_path)
    image_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
    np.random.seed(2017)
    np.random.shuffle(image_paths)
    return image_paths


def run_demo_1(args):
    if args.classifier == "type":
        # Build train/test folder paths
        train_path = os.path.join(TYPE_DATA, TRAIN_FOLDER)
        test_path = os.path.join(TYPE_DATA, TEST_FOLDER)
        # Get train/test file paths
        train_file_paths = get_type_model_files(train_path)
        test_file_paths = get_type_model_files(test_path)
        # Create a type classification model instance
        if args.build:
            model = TypeClassificationModel()
            model.model_path = TYPE_MODEL_PATH
        else:
            model = TypeClassificationModel(TYPE_MODEL_PATH)
        # Run the demo
        DemoClass().run_commands(model, args, train_file_paths, test_file_paths[0:10])
    elif args.classifier == "size":
        # Build train/test folder paths
        train_path = os.path.join(SIZE_DATA, TRAIN_FOLDER, SIZE_CLASSIFICATION_FOLDER)
        test_path = os.path.join(SIZE_DATA, TEST_FOLDER, SIZE_CLASSIFICATION_FOLDER)
        # Get train/test file paths
        train_file_paths = get_size_model_files(train_path)
        test_file_paths = get_size_model_files(test_path)
        # Create a type classification model instance
        if args.build:
            model = SizeClassificationModel()
            model.model_path = SIZE_MODEL_PATH
        else:
            model = SizeClassificationModel(SIZE_MODEL_PATH)
        DemoClass().run_commands(model, args, train_file_paths, test_file_paths[0:10])


""" RUN """
if __name__ == '__main__':
    # USER INTERFACE

    usageStr = """
      USAGE:      python demo_1.py <options>
      EXAMPLES:   (1) python demo_1.py --classifier type --run demo
                    -Runs type classification demonstration
                  (2) python demo_1.py --classifier size --run analyzer
                    -Runs size classification analyzer
                    OR See 'commands.txt' for options to copy and paste
    """

    parser = argparse.ArgumentParser(description=usageStr)

    # For each classification type

    # input: Choose type
    # a) Type classification
    # b) Size classification

    parser.add_argument('-c', '--classifier', nargs="?", choices=["type", "size"], type=str,
                        help="Either `type` or `size`", metavar='CLASSIFIER', default="type")

    # input: Choose testing option

    # a) Demontrate prediction
    # b) Analyze performance

    parser.add_argument('-r', '--run', nargs="?", choices=["demo", "analyzer"], type=str,
                        help="Either `demo` or `analyzer`", metavar='RUN', default="demo")

    # input: Force build
    # No argument, pass -b for True, don't pass for False
    # a) True, if cached model available load model
    # b) False, train new model and cache

    parser.add_argument('-b', '--build', action='store_true')

    # Parse arguments from command line
    # args = parser.parse_args(sys.argv[1:])
    # Example args for testing in console
    # args = parser.parse_args("--classifier size --run demo".split())
    args = parser.parse_args("--classifier type --run demo -b".split())

    run_demo_1(args)

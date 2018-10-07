"""
 Command line script for demo 1 with Goodwill
 Clothing Image Classification Demonstration
 @author Chris Farr 10/4/18
"""
import argparse
import sys
import os

# CONSTANTS
SIZE_CLASSIFICATION_FOLDER = "t_shirt"  # womens_jeans, womens_short_sleeve, womens_long_sleeve
SIZE_DATA = os.path.join("src\\data\\size_data", SIZE_CLASSIFICATION_FOLDER)
TYPE_DATA = "src\\data\\type_data"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"
VALIDATION_FOLDER = "val"

# IMPLEMENTATION
# TODO Implementation Steps:
# 1. Build shells for all of the below classes (consider unit test shells)
# 2. Pull code from FashionClassification repo for model implementation
# 3. Test and build all classes/methods/abstract methods

""" MODEL ABSTRACT IMPLEMENTATION """
# Abstract class:
#   ImageClassificationAbstract
# Abstract Methods:
#   train(image_paths_list)
#       accepts list of image paths, trains model, stores trained model
#   preprocess_images(images_array)
#       accepts images array, return preprocessed images array
#   predict(image_paths_list)
#       accepts list of image paths, returns predicted classes
# Methods:
#   get_images_array(image_paths_list)
#       accepts image paths list, returns images array
#   get_classes_array(image_paths_list)
#       accepts image path, returns image classes
# Child classes:
#   TypeClassificationModel
#   SizeClassificationModel

""" MODEL VALIDATION IMPLEMENTATION """
# ImageClassificationValidation
# Methods
#   cross_validation_summary(model_object, train_folder_path)
#       get file names
#       join file names with train folder path
#       create stratified splits (consider repeated)
#       actual, predicted = [], []
#       loop
#           model_class.train(train_paths)
#           model_class.predict(test_paths)
#           model_class.get_classes_array(test_paths)
#           append actual and predicted
#       np.unique(model_class.get_classes_array(image_paths_list))
#       print_summary(actual, predicted, available_classes)
#   print_summary(actual, predicted, available_classes)
#       # Accuracy, available classes, confusion matrix
#       accuracy_score(actual, predicted)
#       confusion_matrix(actual, predicted, available_classes)

""" DEMO 1 IMPLEMENTATION """
# Base Class:
#   DemoClass
# Methods:
#   run_demo(model_object, train_folder_path, test_folder_path)
#       # train model
#       train_files = os.listdir(train_folder_path)
#       join train_folder_path to create file paths
#       model_object.train(train_file_paths)
#       # create test predictions
#       test_files = os.listdir(test_folder_path)
#       model_object.get_images_array(image_paths_list)
#       model_object.preprocess_images(test_images_array)
#       model_object.predict(test_paths_list)
#       # get actual classes
#       model_object.get_classes_array(test_paths_list)
#       # print model validation summary
#       display_images(orig, preprocessed, actual_label, pred_label)
#   run_analyzer(model_object, train_folder_path)
#       # Perform stratified k-fold prediction
#       # And  print summary of results
#       cross_validation(train_folder_path)
#   display_images(orig, preprocessed, actual_label, pred_label)
#       # loop through arrays
#       # display original image on left, preprocessed image right, overlay label on each image
#       # pressing enter progresses user one image set at a time, esc exits/completes demo
#   run_commands(model_object, args, train_folder_path, test_folder_path)
#       if args.run == "demo":
#           run_demo(model_object, train_folder_path, test_folder_path)
#       elif args.run == "analyzer":
#           run_analyzer(model_object, train_folder_path)

""" DEMO 1 DRIVER """
# Methods:
#   parse_args(sys.argv[1:])
#   run_demo_1(args)
#       if args.classifier == "type":
#           train_path = os.path.join(TYPE_DATA, TRAIN_FOLDER)
#           test_path = os.path.join(TYPE_DATA, TEST_FOLDER)
#           model = TypeClassificationModel()
#           DemoClass().run_commands(model, args, train_path, test_path)
#       elif args.classifier == "size":
#           train_path = os.path.join(SIZE_DATA, TRAIN_FOLDER)
#           test_path = os.path.join(SIZE_DATA, TEST_FOLDER)
#           model = SizeClassificationModel()
#           DemoClass().run_commands(model, args, train_path, test_path)


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

    # Parse arguments from command line
    args = parser.parse_args(sys.argv[1:])
    # Example args
    # args = parser.parse_args("-c type -r demo".split())
    # TODO replace below with driver
    if args.classifier == "type":
        if args.run == "demo":
            print("type", "demo")
            pass
        elif args.run == "analyzer":
            print("type", "analyzer")
            pass
    elif args.classifier == "size":
        if args.run == "demo":
            print("size", "demo")
            pass
        elif args.run == "analyzer":
            print("size", "analyzer")
            pass

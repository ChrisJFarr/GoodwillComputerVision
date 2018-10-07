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

# IMPLEMENTATION

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
#   cross_validation(train_folder_path)
#       get file names
#       create stratified splits (consider repeated)
#       actual, predicted = [], []
#       loop
#           train(train_paths)
#           predict(test_paths)
#           get_classes_array(test_paths)
#           append actual and predicted
#       print_summary(actual, predicted)
#   score(actual, predicted)
#   print_summary(actual, predicted)

""" DEMO 1 IMPLEMENTATION """
# TODO START HERE!!! Nearly complete with sudo code
# Abstract class:
#   DemoOneAbstract
# Abstract methods:
# Methods:
#   run_demo(train_folder_path, test_folder_path)
#       train model
#       train_files = os.listdir(train_folder_path)
#       join train_folder_path to create file paths
#       train(train_files)
#       create test predictions
#       test_files = os.listdir(test_folder_path)
#       get_images_array(image_paths_list)
#       preprocess_image(test_images_array)
#       predict(test_paths_list)
#       get actual classes
#       get_classes_array(test_paths_list)
#       print model validation summary
#       display_images(orig, preprocessed, actual_label, pred_label)
#   run_analyzer(source_folder)
#       Perform stratified k-fold prediction
#       Print summary of results
#       Accuracy, available classes, confusion matrix
#   run_commands(**args)
#       if args.run == "demo":
#           pass
#       elif args.run == "analyzer":
#           pass
#   display_images(orig, preprocessed, actual_label, pred_label)
#       loop through arrays
#       display original image on left, preprocessed image right, overlay label on each image
#       pressing enter progresses user one image set at a time, esc exits/completes demo
# Child classes:
#   TypeDemo
#   SizeDemo

""" DEMO 1 DRIVER """
# Methods:
#   parse_args(sys.argv[1:])
#   run_demo_1(args)
#

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

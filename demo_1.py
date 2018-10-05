"""
 Command line script for demo 1 with Goodwill
 Clothing Image Classification Demonstration
 @author Chris Farr 10/4/18
"""
from optparse import OptionParser

# CONSTANTS
SIZE_CLASSIFICATION_FOLDER = "t_shirt"  # womens_jeans, womens_short_sleeve, womens_long_sleeve

# DEMO 1 DRIVER

# Class that works with model implementations based on inputs from arg parser
# Methods
#   parse_commands
#   run_type_classification_demo
#   run_type_classification_analyzer
#   run_size_classification_demo
#   run_size_classification_analyzer


# USER INTERFACE


def default(str):
    return str + ' [Default: %default]'


usageStr = """
  USAGE:      python demo_1.py <options>
  EXAMPLES:   (1) python demo_1.py --classifier type --run demo
                -Runs type classification demonstration
              (2) python demo_1.py --classifier size --run analyzer
                -Runs size classification analyzer
                OR See 'commands.txt' for options to copy and paste
"""
parser = OptionParser(usageStr)

# For each classification type

# input: Choose type
# a) Type classification
# b) Size classification
parser.add_option('-c', '--classifier', dest='classifier', type='str',
                  help=default('Select either type or size'), metavar='CLASSIFIER', default="type")

# input: Choose testing option

# a) Demontrate prediction
# b) Analyze performance

# Part 1

# Point to a folder with test images
# Loop through showing the original, preprocessed, and prediction
# Show image bounding box

# Part 2

# Perform stratified k-fold prediction
# Print summary of results
# Accuracy, classes, confusion matrix

# IMPLEMENTATION

# Abstract class:
#   ImageClassificationDriver
# Abstract Methods:
#   predict_image_class(image_path)
#   cross_validation_summary(train_folder_path)
#   run_demo(test_folder_path)
#   predict(image)
#   preprocess_image(image)
#   train(train_folder_path)
#   set_classes(classes_map)
#   parse_class(image_path)

# ImageClassificationDriver Children Classes
# TypeClassificationDriver
# SizeClassificationDriver

if __name__ == '__main__':
    args = readCommand(sys.argv[1:])  # Get game components based on input
    run_demo_1(**args)

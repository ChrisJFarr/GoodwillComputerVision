# Command line script for demo


# USER INTERFACE

# For each classification type

# input: Choose type
# a) Type classification
# b) Size classification

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








# ImageClassificationValidation
# Methods
#   cross_validation_summary(train_folder_path, model_class)
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

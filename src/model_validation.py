import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold


class ImageClassificationValidation:
    RANDOM_STATE = 36851234
    REPEATS = 10
    SPLITS = 10

    def cross_validation_summary(self, train_folder_path, model_class):
        actual, predicted, available_classes = cross_validation(train_folder_path, model_class)
        print_summary(actual, predicted, available_classes)

    def cross_validation(self, folder_path, model_class):

        # get file names

        train_image_paths_list = []
        for root, dirs, files in os.walk(train_folder_path + "/train"):
            for name in files:
                train_image_paths_list.append(os.path.join(root, name))

        # create stratified splits (consider repeated)
        rskf = RepeatedStratifiedKFold(n_splits=self.SPLITS, n_repeats=self.REPEATS, random_state=self.RANDOM_STATE)

        actual, predicted = [], []
        # loop
        for train_paths, test_paths in rskf.split(train_image_paths_list)]:
            model_class.train(train_paths)
            predicted.append = model_class.predict(test_paths)
            actual.append = model_class.get_classes_array(test_paths)
        np.unique(model_class.get_classes_array(image_paths_list))
        return actual, predicted, available_classes


    def print_summary(self, actual, predicted, available_classes):
        #to be done for each class?

        print(accuracy_score(actual, predicted))
        
        print(confusion_matrix(actual, predicted))

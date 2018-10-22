import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, RepeatedStratifiedKFold


class ImageClassificationValidation:
    RANDOM_STATE = 36851234
    REPEATS = 10
    SPLITS = 5

    def cross_validation_stratified(self, train_folder_path, model_class):

        # get file names
        train_image_paths_list = []
        train_image_names_list = []
        for root, dirs, files in os.walk("src/data/size_data/train", topdown=False):
            for name in files:
                train_image_paths_list.append(os.path.join(root, name))
                train_image_names_list.append(name)
        #print(train_image_paths_list)
        #print(model_class.get_classes_array(train_image_paths_list))
        # create stratified splits (consider repeated)
        rskf = RepeatedStratifiedKFold(n_splits=self.SPLITS, n_repeats=self.REPEATS, random_state=self.RANDOM_STATE)

        actual, predicted = [], []
        # loop
        for train_paths, test_paths in rskf.split(train_image_paths_list, model_class.get_classes_array(train_image_names_list)):
            model_class.train(train_paths)
            predicted.append = model_class.predict(test_paths)
            actual.append = model_class.get_classes_array(test_paths)
        np.unique(model_class.get_classes_array(image_paths_list))
        return actual, predicted, available_classes


    def print_summary(self, actual, predicted, available_classes):
        #to be done for each class?

        print(accuracy_score(actual, predicted))

        print(confusion_matrix(actual, predicted))

    def cross_validation_summary(self, train_folder_path, model_class):
        actual, predicted, available_classes = self.cross_validation_stratified(train_folder_path, model_class)
        self.print_summary(actual, predicted, available_classes)

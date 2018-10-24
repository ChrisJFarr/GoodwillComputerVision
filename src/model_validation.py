import os
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold
from src.validation_abstract import ValidationAbstract
import pickle


class ImageClassificationValidation(ValidationAbstract):
    RANDOM_STATE = 36851234
    REPEATS = 1
    SPLITS = 10

    def __init__(self, cache_path=None):
        # TODO start here
        pass

    def save(self, actual, predicted, available_classes, cache_path):
        try:
            pickle.dumps((actual, predicted, available_classes), open(cache_path, "wb"))
        except Exception as e:
            print("Unable to store analyzer contents...")
        return

    def load(self, cache_path):
        actual, predicted, available_classes = None, None, None
        try:
            actual, predicted, available_classes = pickle.loads(open(cache_path, "rb"))
        except FileNotFoundError:
            print("Unable to load analyzer contents...")
        return actual, predicted, available_classes

    def cross_validation_stratified(self, train_folder_path, model_class):
        # create stratified splits (consider repeated)
        rskf = RepeatedStratifiedKFold(n_splits=self.SPLITS, n_repeats=self.REPEATS, random_state=self.RANDOM_STATE)
        train_image_names_list = [os.path.basename(image_path) for image_path in train_folder_path]
        actual, predicted = [], []
        # loop
        for train_paths, test_paths in rskf.split(train_folder_path,
                                                  model_class.get_classes_array(train_image_names_list)):
            x_train = list(train_folder_path[i] for i in train_paths)
            x_test = list(train_folder_path[i] for i in test_paths)
            model_class.train(x_train)
            predicted.extend(model_class.predict(x_test))
            test_image_names_list = [os.path.basename(image_path) for image_path in x_test]
            actual.extend(model_class.get_classes_array(test_image_names_list))
        available_classes = np.unique(model_class.get_classes_array(train_image_names_list))

        return actual, predicted, available_classes

    @staticmethod
    def plot_metrics(cvacc, cm, classes,
                     normalize=False,
                     title='Confusion matrix',
                     cmap=plt.cm.Blues):

        output_message = "Cross validated accuracy: %.1f" % cvacc + "%"
        print(output_message, "")

        print('Confusion matrix, without normalization')
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return

    def print_summary(self, actual, predicted, available_classes):
        cross_validated_accuracy = accuracy_score(actual, predicted)
        conf_matrix = confusion_matrix(actual, predicted)
        np.set_printoptions(precision=2)

        # Plot metrics
        plt.figure()
        self.plot_metrics(cross_validated_accuracy * 100, conf_matrix, classes=available_classes,
                          title='Confusion matrix')
        plt.show()
        return

    def cross_validation_summary(self, train_folder_path, model_class, cache_path=None):
        actual, predicted, available_classes = None, None, None
        if cache_path is not None:
            actual, predicted, available_classes = self.load(cache_path)
        if actual is None or predicted is None or available_classes is None:
            actual, predicted, available_classes = self.cross_validation_stratified(train_folder_path, model_class)
        self.print_summary(actual, predicted, available_classes)
        return

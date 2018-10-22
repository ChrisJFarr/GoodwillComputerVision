from src.models.model_abstract import ImageClassificationAbstract
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import LinearSVC
import os


TARGET_SIZE = (120, 180)
SIZE_MAP = {
    "3xl": "3xl-large", "2xl": "3xl-large", "xlarge": "3xl-large", "large": "3xl-large",
    "medium": "medium-xsmall", "small": "medium-xsmall", "xsmall": "medium-xsmall"
}
# 3xl-large, medium-xsmall
AUGMENT = True
AUGMENTED_SIZE = 500


class SizeClassificationModel(ImageClassificationAbstract):

    def __init__(self, *args, **kwargs):
        ImageClassificationAbstract.__init__(self, *args, **kwargs)

    # Override Abstract Methods:
    @staticmethod
    def get_classes_array(image_names_list):
        # accepts image path, returns image classes
        classes = []
<<<<<<< HEAD
        for file_name in image_paths_list:
            print(file_name.split("_")[1])
=======
        for file_name in image_names_list:
>>>>>>> master
            classes.append(file_name.split("_")[1])
        classes = [SIZE_MAP.get(sz.lower()) for sz in classes]
        return np.array(classes)

    def train(self, image_paths_list):
        # accepts list of image paths, trains model, stores trained model
        # Load images
        images_array = self.get_images_array(image_paths_list)
        # Preprocess x_data
        x_data = self.preprocess_images(images_array)
        # Extract y_data
        # Get file names from image paths
        file_names = [os.path.basename(image_path) for image_path in image_paths_list]
        y_data = self.get_classes_array(file_names)
        # Augment data
        if AUGMENT:
            x_data, y_data = self.created_augmented_data(x_data, y_data)
        # Flatten and scale image data
        x_data = np.array([img.flatten() / 255. for img in x_data])
        # Train model
        model = LinearSVC(random_state=0, class_weight="balanced")
        model.fit(x_data, y_data)
        # Store trained model
        self.set_model(model)
        return None

    def predict(self, image_paths_list):
        # accepts list of image paths, returns predicted classes
        # Load images
        images_array = self.get_images_array(image_paths_list)
        # Preprocess x_data
        x_data = self.preprocess_images(images_array)
        # Flatten and scale image data
        x_data = np.array([img.flatten() / 255. for img in x_data])
        # Get predictions
        predictions = self.get_model().predict(x_data)
        return predictions

    @staticmethod
    def preprocess_images(images_array):
        image_list = list(images_array)
        for i in range(len(images_array)):
            image = image_list[i]
            # accepts images array, return preprocessed images array
            image = cv2.resize(image, TARGET_SIZE)
            # Convert to RGB colorspace
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # blur the image
            kernel = np.ones((3, 3), np.float32) / 5
            image = cv2.filter2D(image, -1, kernel)
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Perform Canny edge detection
            image = cv2.Canny(image, 100, 200)
            image_list[i] = image
        return np.array(image_list)

    @staticmethod
    def created_augmented_data(x_data, y_data):
        # Generate more data
        x_data = np.array([np.expand_dims(image, 2) for image in x_data])
        datagen = ImageDataGenerator(
            rotation_range=1,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
        )
        augmented_x_data = []
        augmented_y_data = []
        num_augmented = 0
        for x_batch, y_batch in datagen.flow(x_data, y_data, batch_size=1, shuffle=False):
            augmented_x_data.append(x_batch[0])
            augmented_y_data.append(y_batch[0])
            num_augmented += 1
            if num_augmented >= AUGMENTED_SIZE:
                break
        augmented_x_data = np.array(augmented_x_data)
        augmented_y_data = np.array(augmented_y_data)
        return augmented_x_data, augmented_y_data

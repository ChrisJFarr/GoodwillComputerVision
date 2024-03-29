from src.models.model_abstract import ImageClassificationAbstract
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from scipy.stats import itemfreq
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler

TARGET_SIZE = (100, 150)
AUGMENT = False
AUGMENTED_SIZE = 20


class TypeClassificationModel(ImageClassificationAbstract):

    def __init__(self):
        ImageClassificationAbstract.__init__(self)
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))

    # Override Abstract Methods:
    @staticmethod
    def get_classes_array(image_names_list):
        # accepts image path, returns image classes
        classes = []
        for file_name in image_names_list:
            classes.append(file_name.split("_")[0])
        return np.array(classes)

    def train(self, image_paths_list):
        # accepts list of image paths, trains model, stores trained model
        # Load images
        images_array = self.get_images_array(image_paths_list)
        # Preprocess x_data
        x_data = self.preprocess_images(images_array)
        # Extract y_data
        # Get file names from image paths
        file_names = [image_path.split("\\")[-1] for image_path in image_paths_list]
        y_data = self.get_classes_array(file_names)
        # Augment data
        if AUGMENT:
            x_data, y_data = self.created_augmented_data(x_data, y_data)
        # Randomly shuffle
        rand_i = list(range(len(x_data)))
        np.random.shuffle(rand_i)
        x_data, y_data = np.array([x_data[i] for i in rand_i]), [y_data[i] for i in rand_i]
        # Get hog features
        x_data = list(x_data)
        features = [self.hog_feature_extractor(image) for image in x_data]
        # Get LBP features
        # features.extend([self.lbp_feature_feature_extractor(image) for image in x_data])
        # Convert back to array
        x_data = np.array(features)
        print("x_data.shape", x_data.shape)
        print("x_data", x_data)
        assert len(x_data.shape) > 1, "Mismatching features lengths"
        # Scale features
        # x_data = self.x_scaler.fit_transform(x_data)
        # Train model
        model = LinearSVC(random_state=0, class_weight="balanced")
        model.fit(x_data, y_data)
        # Store trained model
        self.set_model(model)
        return None

    def predict(self, image_paths_list):
        # accepts list of image paths, returns predicted classes
        images_array = self.get_images_array(image_paths_list)
        # Preprocess x_data
        x_data = self.preprocess_images(images_array)
        # Flatten and scale image data
        x_data = self.x_scaler.transform(x_data)
        # Get predictions
        predictions = self.get_model().predict(x_data)
        return predictions

    @staticmethod
    def preprocess_images(images_array):
        # accepts images array, return preprocessed images array
        image_list = list(images_array)
        for i in range(len(images_array)):
            image = image_list[i]
            # accepts images array, return preprocessed images array
            image = cv2.resize(image, TARGET_SIZE)
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_list[i] = image
        return np.array(images_array)

    @staticmethod
    def hog_feature_extractor(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_feature = hog(image, orientations=8, pixels_per_cell=(32, 32),
                          cells_per_block=(2, 2))
        return hog_feature

    @staticmethod
    def lbp_feature_feature_extractor(image):
        num_row = 6
        num_col = 4
        radius = 3
        no_points = 8 * radius
        lbp = []
        img_size = image.shape
        row_size = img_size[0] // num_row
        col_size = img_size[1] // num_col
        for row in range(1, row_size * num_row, row_size):
            for col in range(1, int(col_size * num_col), col_size):
                # Extracting blocks and generating features
                if (row == 1) and (col == 1):
                    continue
                if (row == 1) and (col == (col_size * num_col - col_size + 1)):
                    continue
                block_r = image[row:(row + row_size - 1), col:(col + col_size - 1), 0]
                block_g = image[row:(row + row_size - 1), col:(col + col_size - 1), 1]
                block_b = image[row:(row + row_size - 1), col:(col + col_size - 1), 2]
                lbp_temp_r = local_binary_pattern(block_r, no_points, radius, method='uniform')
                lbp_temp_g = local_binary_pattern(block_g, no_points, radius, method='uniform')
                lbp_temp_b = local_binary_pattern(block_b, no_points, radius, method='uniform')
                # Calculate the histogram
                x_r = itemfreq(lbp_temp_r.ravel())
                x_g = itemfreq(lbp_temp_g.ravel())
                x_b = itemfreq(lbp_temp_b.ravel())
                # Normalize the histogram
                hist_r = x_r[:, 1] / sum(x_r[:, 1])
                hist_g = x_g[:, 1] / sum(x_g[:, 1])
                hist_b = x_b[:, 1] / sum(x_b[:, 1])
                lbp.extend(hist_r)
                lbp.extend(hist_g)
                lbp.extend(hist_b)
        return lbp

    @staticmethod
    def created_augmented_data(x_data, y_data):
        # Generate more data
        datagen = ImageDataGenerator(
            # TODO Add more augmentation (change shape)
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

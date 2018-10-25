from src.models.model_abstract import ImageClassificationAbstract
import cv2
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from scipy.stats import itemfreq
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import LinearSVC
import os

TARGET_SIZE = (400, 600)
AUGMENTED_SIZE = 200


class TypeClassificationModel(ImageClassificationAbstract):

    def __init__(self, *args, **kwargs):
        ImageClassificationAbstract.__init__(self, *args, **kwargs)

    # Override Abstract Methods:
    def train(self, image_paths_list):
        # accepts list of image paths, trains model, stores trained model
        x_data, y_data = self.get_x_y_data(image_paths_list, augment=False)
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
        x_data, y_data = self.get_x_y_data(image_paths_list, augment=False)
        # Get predictions
        predictions = self.get_model().predict(x_data)
        return predictions

    def get_x_y_data(self, image_paths_list, augment=False):
        # Load images
        images_array = self.get_images_array(image_paths_list)
        # Preprocess x_data
        x_data = self.preprocess_images(images_array)
        # Extract y_data
        # Get file names from image paths
        file_names = [os.path.basename(image_path) for image_path in image_paths_list]
        # file_names = [image_path.split("/")[-1] for image_path in image_paths_list]
        y_data = self.get_classes_array(file_names)
        # Augment data
        if augment:
            x_data, y_data = self.created_augmented_data(x_data, y_data)
        # Get features
        features_list = []
        for image in x_data:
            hog_features = self.hog_feature_extractor(image)
            # Get LBP features
            lbp_features = self.lbp_feature_feature_extractor(image)
            # Combine features
            features = []
            features.extend(hog_features)
            features.extend(lbp_features)
            features_list.append(features)
        # Convert features_list to array
        x_data = np.array(features_list)
        assert len(x_data.shape) > 1, "Mismatching features lengths: %s" % [len(x) for x in x_data]
        return x_data, y_data

    @staticmethod
    def get_classes_array(image_names_list):
        # accepts image path, returns image classes
        classes = []
        for file_name in image_names_list:
            classes.append(file_name.split("_")[0])
        return np.array(classes)

    @staticmethod
    def preprocess_images(images_array):
        # accepts images array, return preprocessed images array
        image_list = list(images_array)
        for i in range(len(images_array)):
            image = image_list[i]
            # # accepts images array, return preprocessed images array
            image = cv2.resize(image, TARGET_SIZE)
            # # Experimental image enhancements
            # # https://chrisalbon.com/machine_learning/preprocessing_images/enhance_contrast_of_color_image/
            # # Convert to YUV
            # # image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # # Apply histogram equalization
            # # image[:, :, 0] = cv2.equalizeHist(image[:, :, 0])
            # # Convert to BGR
            # # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
            # # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
            # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # l, a, b = cv2.split(lab)
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            # cl = clahe.apply(l)
            # limg = cv2.merge((cl, a, b))
            # image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

            # Crop to only the object
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
            thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            cont_img = closing.copy()
            _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x))
            rect = cv2.minAreaRect(sorted_contours[-1])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x1 = max(min(box[:, 0]), 0)
            y1 = max(min(box[:, 1]), 0)
            x2 = max(max(box[:, 0]), 0)
            y2 = max(max(box[:, 1]), 0)

            # Enhance
            image_cropped = image[y1:y2, x1:x2]
            lab = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            image_cropped = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # Use fill
            image[y1:y2, x1:x2] = image_cropped
            # image = cv2.resize(image_cropped, TARGET_SIZE)

            image_list[i] = image
        return np.array(image_list)

    @staticmethod
    def hog_feature_extractor(image):
        im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_feature = hog(im_gray, orientations=8, pixels_per_cell=(32, 32),
                          cells_per_block=(2, 2), block_norm="L1")  # L2-Hys
        return hog_feature

    @staticmethod
    def lbp_feature_feature_extractor(image):

        def normalize_lbp_counts(lbp_counts):
            counts_dict = dict(zip(*lbp_counts))
            for i in range(0, no_points + 2):
                if counts_dict.get(i) is None:
                    counts_dict[i] = 0
            return counts_dict

        num_row = 10
        num_col = 5
        radius = 3
        no_points = 8 * radius
        lbp = []
        img_size = image.shape
        # print("image_size", img_size)
        row_size = img_size[0] // num_row
        col_size = img_size[1] // num_col
        for row in list(range(0, row_size * num_row, row_size)):
            for col in list(range(0, int(col_size * num_col), col_size)):
                # Extracting blocks and generating features
                if (row == 0) and (col == 0):
                    continue
                if (row == 0) and (col == (col_size * num_col - col_size)):
                    continue

                block_r = image[row:(row + row_size), col:(col + col_size), 0]
                block_g = image[row:(row + row_size), col:(col + col_size), 1]
                block_b = image[row:(row + row_size), col:(col + col_size), 2]
                lbp_temp_r = local_binary_pattern(block_r, no_points, radius, method='uniform')
                lbp_temp_g = local_binary_pattern(block_g, no_points, radius, method='uniform')
                lbp_temp_b = local_binary_pattern(block_b, no_points, radius, method='uniform')
                # Calculate the histogram
                x_r = np.unique(lbp_temp_r.ravel(), return_counts=True)
                x_g = np.unique(lbp_temp_g.ravel(), return_counts=True)
                x_b = np.unique(lbp_temp_b.ravel(), return_counts=True)

                # Align the bins
                x_r_t = normalize_lbp_counts(x_r)
                x_g_t = normalize_lbp_counts(x_g)
                x_b_t = normalize_lbp_counts(x_b)

                x_r_counts = [value for key, value in sorted(x_r_t.items())]
                x_g_counts = [value for key, value in sorted(x_g_t.items())]
                x_b_counts = [value for key, value in sorted(x_b_t.items())]

                # Normalize the histogram
                hist_r = x_r_counts / sum(x_r_counts)
                hist_g = x_g_counts / sum(x_g_counts)
                hist_b = x_b_counts / sum(x_b_counts)
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

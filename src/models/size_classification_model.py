from src.models.model_abstract import ImageClassificationAbstract


class SizeClassificationModel(ImageClassificationAbstract):

    def __init__(self):
        ImageClassificationAbstract.__init__(self)

    # Override Abstract Methods:
    def train(self, image_paths_list):
        # accepts list of image paths, trains model, stores trained model
        return None

    @staticmethod
    def preprocess_images(images_array):
        # accepts images array, return preprocessed images array
        return images_array

    def predict(self, image_paths_list):
        # accepts list of image paths, returns predicted classes
        return image_paths_list


# Abstract class:
#   ImageClassificationAbstract
# Abstract Methods:
#   train(image_paths_list)
#       accepts list of image paths, trains model, stores trained model
#   preprocess_images(images_array)
#       accepts images array, return preprocessed images array
#   predict(image_paths_list)
#       accepts list of image paths, returns predicted classes
# Methods:
#   get_images_array(image_paths_list)
#       accepts image paths list, returns images array
#   get_classes_array(image_paths_list)
#       accepts image path, returns image classes

import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from matplotlib import pyplot as plt

source_data = "src/data/type_data/train"

# Get list of training files
file_paths = []
for folder in os.listdir(source_data):
    for file_name in os.listdir(os.path.join(source_data, folder)):
        file_paths.append(os.path.join(source_data, folder, file_name).replace("\\", "/"))
np.random.seed(100)
np.random.shuffle(file_paths)

# Get list of images
image_list = [cv2.imread(file_path) for file_path in file_paths]
image_list = [im for im in image_list if im is not None]
target_list = [nm.split("/")[-1].split("_")[0] for nm in file_paths]

# Resize images
image_list = [cv2.resize(im, (120, 180)) for im in image_list]

# Train/test/validation split
test_size, val_size = int(len(image_list) * .10), int(len(image_list) * .10)
train_images = image_list[test_size + val_size:]
val_images = image_list[test_size:test_size + val_size]
test_images = image_list[:test_size]

train_labels = target_list[test_size + val_size:]
val_labels = target_list[test_size:test_size + val_size]
test_labels = target_list[:test_size]

assert all((len(train_images) == len(train_labels),
            len(val_images) == len(val_labels),
            len(test_images) == len(test_labels)))


# Augment train to create more data
def created_augmented_data(x_data, y_data, n=1000):
    # x_data_reshaped = np.array([np.expand_dims(image, 2) for image in x_data])
    # x_data_reshaped.shape
    # Generate more data
    datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05,
    )
    augmented_x_data = []
    augmented_y_data = []
    num_augmented = 0
    for x_batch, y_batch in datagen.flow(x_data, y_data, batch_size=1, shuffle=False):
        augmented_x_data.append(x_batch[0])
        augmented_y_data.append(y_batch[0])
        num_augmented += 1
        if num_augmented >= n:
            break
    augmented_x_data = np.array(augmented_x_data).astype(np.uint8)
    return augmented_x_data, augmented_y_data


x_train, train_labels = created_augmented_data(np.array(train_images), np.array(train_labels), n=200)
x_valid, val_labels = np.array(val_images), np.array(val_labels)
x_test, test_labels = np.array(test_images), np.array(test_labels)


# for i in range(10):
#     cv2.imshow("test", x_train[i].astype(np.uint8))
#     cv2.waitKey()
#     cv2.destroyAllWindows()


# Preprocess images with canny edge detection
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    image = cv2.Canny(image, 50, 100)
    return image

# i = np.random.randint(0, len(image_list))
# plt.imshow(preprocess_image(image_list[i].astype(np.uint8)), cmap="gray")
#
# i = np.random.randint(0, len(image_list))
# image = image_list[i].astype(np.uint8)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# # image = cv2.GaussianBlur(image, (7, 7), 0)
# image = cv2.Canny(image, 50, 100)
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
#
# plt.imshow(image, cmap="gray")


x_train = np.array([preprocess_image(img) for img in list(x_train)])
x_valid = np.array([preprocess_image(img) for img in list(x_valid)])
x_test = np.array([preprocess_image(img) for img in list(x_test)])

x_train = np.expand_dims(x_train, 3)
x_valid = np.expand_dims(x_valid, 3)
x_test = np.expand_dims(x_test, 3)

from keras.utils import np_utils
from pandas import get_dummies

y_train = get_dummies(train_labels).values
y_valid = get_dummies(val_labels).values
y_test = get_dummies(test_labels).values

# Train/validate CNN
# Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.layers import Dropout, Flatten, Dense
from keras import Sequential, optimizers, layers
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint

model = Sequential()
# TODO: Define your architecture.
model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu',
                 input_shape=x_train.shape[1:]))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.summary()


model.compile(optimizer=optimizers.Adamax(), loss='categorical_crossentropy', metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint, EarlyStopping

# TODO: specify the number of epochs that you would like to use to train the model.

epochs = 100

checkpointer = ModelCheckpoint(filepath='dev/saved_models/weights.best.type.1.0.hdf5',
                               verbose=1, save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')

hist = model.fit(x_train, y_train, batch_size=32, epochs=epochs,
                 validation_data=(x_valid, y_valid), callbacks=[checkpointer, early_stopping],
                 verbose=1, shuffle=True)

# # Analyze test performance
# dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#
# # report test accuracy
# test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
# print('Test accuracy: %.4f%%' % test_accuracy)
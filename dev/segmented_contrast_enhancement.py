import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

# Pull in test images
folder_path = "src/data/type_data/train/womens_dresses"
file_names = os.listdir(folder_path)
file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
len(file_paths)
# Segment image


# Bounding box?
# Background?

# Enhance


# http://blog.christianperone.com/2014/06/simple-and-effective-coin-segmentation-using-python-and-opencv/
i = 0

# Find the aligned rectangle
i += 1
image = cv2.imread(file_paths[i])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
cont_img = closing.copy()
_, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x))
rect = cv2.minAreaRect(sorted_contours[-1])
# rect = cv2.boundingRect(sorted_contours[-1])
box = cv2.boxPoints(rect)
box = np.int0(box)
# image[box[0][0]]
# cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

x1 = max(min(box[:, 0]), 0)
y1 = max(min(box[:, 1]), 0)
x2 = max(max(box[:, 0]), 0)
y2 = max(max(box[:, 1]), 0)

# Enhance only within the box
image_cropped = image[y1:y2, x1:x2]
lab = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
image_cropped = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
image[y1:y2, x1:x2] = image_cropped
# View
plt.imshow(image)

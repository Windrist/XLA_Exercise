import cv2
import os
import numpy as np


def update_mean(image, y, x):
    sum_image = 0
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            sum_image += image[i][j]
    mean = int(sum_image / 9)
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            image[i][j] = mean
    return image


def update_median(image, y, x):
    pixel_list = []
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            pixel_list.append(image[i][j])
    pixel_list.sort()
    image[y][x] = pixel_list[4]
    return image


class BT(object):

    # Init Variables
    def __init__(self):
        self.import_dir = "Input/"
        self.input_rgb = []
        self.input_gray = []
        self.test_list = []

        # Get Images From Input Folder
        self.get_image()

        # Covert All Images to Grayscale
        self.cvt_gray()

        # Show Gray Images and Save
        for gray_img in self.input_gray:
            cv2.imshow('Test', gray_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite('Output/Gray.png', gray_img)

    # Import Images
    def get_image(self):
        list_image = os.listdir(self.import_dir)
        for file in list_image:
            self.input_rgb.append(cv2.imread(self.import_dir + file))

    # Function to Convert RGB Images to Gray Images
    def cvt_gray(self):
        for image in self.input_rgb:
            gray = np.rint(np.dot(image[..., :3], [0.114, 0.587, 0.299]))
            gray = gray.astype(np.uint8)
            self.input_gray.append(gray)

    # Main Code for Invert Gray Images
    def inverse_gray(self):
        inv_gray = []
        for gray in self.input_gray:
            gray = np.absolute(255 - gray)
            inv_gray.append(gray)
        return inv_gray

    # Main Code for Mean Filter
    def mean_filter(self):
        mean_image = []
        for image in self.input_gray:
            lcol, lrow = image.shape
            for i in range(1, lcol-1):
                for j in range(1, lrow-1):
                    image = update_mean(image, i, j)
            mean_image.append(image)
        return mean_image

    # Main Code for Median Filter
    def median_filter(self):
        median_image = []
        for image in self.input_gray:
            lcol, lrow = image.shape
            for i in range(1, lcol-1):
                for j in range(1, lrow-1):
                    image = update_median(image, i, j)
            median_image.append(image)
        return median_image








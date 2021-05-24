import cv2
import os
import numpy as np
import math
import matplotlib.pyplot as plt

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


def update_sobel(image, y, x):
    Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    S1 = 0
    S2 = 0

    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            S1 += Gx[i-(y-1)][j-(x-1)] * image[i][j]
            S2 += Gy[i-(y-1)][j-(x-1)] * image[i][j]
    return np.sqrt(S1*S1 + S2*S2)


def update_laplacian(image, y, x):
    M1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    M2 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    M3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    L1 = 0
    L2 = 0

    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            L1 += M1[i-(y-1)][j-(x-1)] * image[i][j]
            L2 += M1[i-(y-1)][j-(x-1)] * image[i][j]
    return L1 + L2


def get_log(sigma):
    mask_length = math.ceil(float(3) * float(sigma))
    if(mask_length % 2 == 0):
        mask_length = mask_length + 1
    M = np.zeros((mask_length, mask_length))

    for i in range(0, mask_length):
        for j in range(0, mask_length):
            nom = j**2 + i**2 - 2*(sigma**2)
            denom = 2 * math.pi * (sigma**6)
            expo = math.exp(-(i**2 + j**2) / (2*(sigma**2)))
            M[i][j] = nom * expo / denom
    return M


def update_laplacian_gaussian(image, y, x, M, mcol):
    L1 = 0
    L2 = 0

    for i in range(y - mcol, y + mcol + 1):
        for j in range(x - mcol, x + mcol + 1):
            L1 += M[i-(y-1)][j-(x-1)] * image[i][j]
            L2 += M[i-(y-1)][j-(x-1)] * image[i][j]
    return L1 + L2


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
    
    # Main Code for Calculate Histogram
    def calc_histogram(self):
        histr_image = []
        for image in self.input_gray:
            histogram = []
            histr = np.linspace(0, 255, 256)
            for k in range(256):
                mask = image[image == histr[k]]
                histogram.append(len(mask))
            histr_image.append(histogram)
        return histr_image
    
    # Main Code for Gradient Sobel in Detect Edge Application
    def edge_sobel(self):
        edge_image = []
        for image in self.input_gray:
            lcol, lrow = image.shape
            output_image = np.zeros((lcol, lrow))
            for i in range(1, lcol - 1):
                for j in range(1, lrow - 1):
                    output_image[i][j] = update_sobel(image, i, j)
            edge_image.append(output_image)
        return edge_image

    # Main Code for Laplacian in Detect Edge Application
    def edge_laplacian(self):
        edge_image = []
        for image in self.input_gray:
            lcol, lrow = image.shape
            output_image = np.zeros((lcol, lrow))
            for i in range(1, lcol - 1):
                for j in range(1, lrow - 1):
                    output_image[i][j] = update_laplacian(image, i, j)
            edge_image.append(output_image)
        return edge_image

    # Main Code for Laplacian of Gaussian in Detect Edge Application 
    def edge_laplacian_gaussian(self, sigma):
        edge_image = []
        for image in self.input_gray:
            lcol, lrow = image.shape
            M = get_log(sigma)
            mcol = int(math.floor(M.shape[0]/2))
            output_image = np.zeros((lcol, lrow))
            for i in range(mcol, lcol - mcol):
                for j in range(mcol, lrow - mcol):
                    output_image[i][j] = update_laplacian_gaussian(image, i, j, M, mcol)
            edge_image.append(output_image)
        return edge_image
    
    # Main Code for Fourier Transform
    def fft(self):
        lpf_image = []
        for image in self.input_gray:
            float_img = np.float32(image)
            dft_image = cv2.dft(float_img, flags = cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft_image)

            magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0],dft_shift[:, :, 1]))

            plt.subplot(121),plt.imshow(image, cmap = 'gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
            plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
            plt.show()

            rows, cols = image.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols, 2), np.uint8)
            mask[crow - int(0.1*crow):crow + int(0.1*crow), ccol - int(0.1*ccol):ccol + int(0.1*ccol)] = 0
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

            plt.subplot(121),plt.imshow(image, cmap = 'gray')
            plt.title('Input Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
            plt.title('Output Image after Low Pass Filter'), plt.xticks([]), plt.yticks([])
            plt.show()
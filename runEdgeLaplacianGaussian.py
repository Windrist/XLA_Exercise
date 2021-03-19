import cv2
from filterTools import BT

BT6 = BT()

sigma = 2

for gray_img in BT6.edge_laplacian_gaussian(sigma):
    cv2.imshow('Test', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output/Laplacian Edge Gaussian.png', gray_img)
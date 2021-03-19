import cv2
from filterTools import BT

BT5 = BT()

for gray_img in BT5.edge_sobel():
    cv2.imshow('Test', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output/Sobel Edge.png', gray_img)
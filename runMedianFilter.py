import cv2
from filterTools import BT

BT3 = BT()

for gray_img in BT3.median_filter():
    cv2.imshow('Test', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output/Median Filter.png', gray_img)
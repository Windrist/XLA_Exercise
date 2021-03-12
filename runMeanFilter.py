import cv2
from filterTools import BT

BT2 = BT()

for gray_img in BT2.mean_filter():
    cv2.imshow('Test', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output/Mean Filter.png', gray_img)
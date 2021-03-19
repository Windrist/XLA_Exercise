import cv2
from filterTools import BT

BT1 = BT()

for inv_gray_img in BT1.inverse_gray():
    cv2.imshow('Test', inv_gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output/Inverse Gray.png', inv_gray_img)

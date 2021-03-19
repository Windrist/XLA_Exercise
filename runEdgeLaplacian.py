import cv2
from filterTools import BT

BT6 = BT()

for gray_img in BT6.edge_laplacian():
    cv2.imshow('Test', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Output/Laplacian Edge.png', gray_img)
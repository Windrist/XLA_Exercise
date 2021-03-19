import cv2
import matplotlib.pyplot as plt
from filterTools import BT

BT4 = BT()

for histr in BT4.calc_histogram():
    plt.plot(histr)
    plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterTools import BT

BT4 = BT()

for histr in BT4.calc_histogram():
    x_axis = np.linspace(0, 255, 256)

    plt.bar(x_axis, histr, width=1)
    plt.show()
import cv2
import os
import numpy as np
import os, glob
from numpy import linalg as LA

np.set_printoptions(threshold= np.inf)
np.set_printoptions(suppress=True)
os.chdir('fish_data')
os.chdir('Trout')

img = cv2.imread("00001.png")
print(img.shape)

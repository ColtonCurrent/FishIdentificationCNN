import cv2
from keras import models
import os
import tensorflow as tf
import numpy as np
import os, glob
from numpy import linalg as LA

np.set_printoptions(threshold= np.inf)
np.set_printoptions(suppress=True)

#Load Model and Checkpoints#
model = models.load_model('fishmodel.h5')
model.load_weights(tf.train.latest_checkpoint("checkpoints"))

#Image Preprocessing and Prediction#
os.chdir("fish_data")
os.chdir("Trout")
img = cv2.imread("00001.png")
img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
networkOutput = model.predict(img_array)

#Label and Display Image#
Labels = ["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout"]
classLabel = np.argmax(networkOutput)

cv2.imshow(Labels[classLabel], img)
cv2.waitKey(0)

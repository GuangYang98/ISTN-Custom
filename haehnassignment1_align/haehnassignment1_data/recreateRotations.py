import pickle
import cv2
import os
import numpy as np 
import scipy.ndimage as nd
import random

def nameImage(num):
    x = "mask0"
    if (num < 100):
        x = x + "0"
        if (num < 10):
            x = x + "0"
    x = x + str(num) + ".png"

    return x

#test = cv2.imread("mask0000.png")
#print(test.shape)

img = np.copy(np.pad(cv2.imread("mask0000.png"), ((256,256), (128,128), (0,0)), mode='constant', constant_values=0))

rotations = pickle.load(open("rotations.dat","rb"))

i = 0
for t in rotations:
    i = i + 1
    r_img = nd.rotate(img, t, reshape=False)
    cv2.imwrite(nameImage(i), r_img)


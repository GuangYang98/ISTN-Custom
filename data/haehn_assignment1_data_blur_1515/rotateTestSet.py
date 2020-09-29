import cv2
import scipy.ndimage as nd
import random

import numpy as np
from PIL import Image
import pickle
import os

# os.mkdir("Test_Images")
# os.mkdir("Test_Masks")
# os.mkdir("Train_Images")
# os.mkdir("Train_Masks")

# os.mkdir("Test_Images_Star")
# os.mkdir("Test_Masks_Star")
# os.mkdir("Train_Images_Star")
# os.mkdir("Train_Masks_Star")

os.mkdir("Test_Images_Prime")
os.mkdir("Test_Masks_Prime")

SUBDIR = "istn/data/lucci_128_rotated"

rotations = {}
rotations['Test_Rotations'] = []
rotations['Train_Rotations'] = []


IMAGESHAPE = (88,88)
PADSHAPE = ((20,20), (20,20), (0,0))
MAXANGLE = 60

INPUT_IMAGES_FOLDER =  "Test_Images"
INPUT_MASKS_FOLDER = "Test_Masks"

def rotate(name, img, angle):
    print("reshaped: " + name)
    # res = cv2.resize(img, dsize=IMAGESHAPE, interpolation=cv2.INTER_AREA)
    # pad = np.copy(np.pad(res, PADSHAPE, mode='constant', constant_values=0))
    rot = np.copy(nd.rotate(img, angle, reshape=False))
    return rot

def resizeAndPad( img):
    res = cv2.resize(img, dsize=IMAGESHAPE, interpolation=cv2.INTER_AREA)
    pad = np.copy(np.pad(res, PADSHAPE, mode='constant', constant_values=0))
    return pad

def zeroPadName(i):
    x = ""
    if (i < 1000):
        x = x + "0"
    if (i < 100):
        x = x + "0"
    if (i < 10):
        x = x + "0"
    x = x + str(i)
    return x

def readRotateOutputImage(inputFolder, rotatedOutputFolder, inputImgName, angle, numImg):
    inputPath = os.path.join(inputFolder, inputImgName)
    inputImg = cv2.imread(inputPath)
    reshapedImg = rotate(inputPath, inputImg, angle)

    rotatedOutputImg = Image.fromarray(reshapedImg)
    outputImgName = str(numImg) + ".png"
    outputPath = os.path.join(rotatedOutputFolder, outputImgName) 
    print("to: " + outputPath)
    rotatedOutputImg.save(outputPath)

for i in range(0, 60):
    index = i
    angle = MAXANGLE * random.random()

    inputFolder = INPUT_IMAGES_FOLDER
    rotatedOutputFolder = "Test_Images_Prime"
    inputImgName = str(i) + ".png"
    readRotateOutputImage(inputFolder, rotatedOutputFolder, inputImgName, angle, index)

    inputFolder = INPUT_MASKS_FOLDER
    rotatedOutputFolder = "Test_Masks_Prime"
    inputImgName = str(i) + ".png"
    readRotateOutputImage(inputFolder, rotatedOutputFolder, inputImgName, angle, index)


# pickle.dump(rotations, open('rotations.dat', "wb"))

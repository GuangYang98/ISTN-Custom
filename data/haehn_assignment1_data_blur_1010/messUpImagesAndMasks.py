import cv2
import scipy.ndimage as nd
import random

import numpy as np
from PIL import Image
import pickle
import os

os.mkdir("Test_Images")
os.mkdir("Test_Masks")
os.mkdir("Train_Images")
os.mkdir("Train_Masks")
#
os.mkdir("Test_Images_Unaligned")
os.mkdir("Test_Masks_Unaligned")
os.mkdir("Train_Images_Unaligned")
os.mkdir("Train_Masks_Unaligned")

rotations = {}
rotations['Test_Rotations'] = []
rotations['Train_Rotations'] = []


IMAGESHAPE = (88, 88)
PADSHAPE = ((20,20), (20,20), (0,0))
MAXANGLE = 60

INPUT_IMAGES_FOLDER = "Images"
INPUT_MASKS_FOLDER = "Masks"

def reShape(name, img, angle, blur):
    print("reshaped: " + name)
    res = cv2.resize(img, dsize=IMAGESHAPE, interpolation=cv2.INTER_AREA)
    pad = np.copy(np.pad(res, PADSHAPE, mode='constant', constant_values=0))
    rot = nd.rotate(pad, angle, reshape=False)
    if (blur == True):
        rot = cv2.blur(rot, (10, 10))
    return rot

def resizeAndPad(img, blur):
    res = cv2.resize(img, dsize=IMAGESHAPE, interpolation=cv2.INTER_AREA)
    pad = np.copy(np.pad(res, PADSHAPE, mode='constant', constant_values=0))
    if (blur == True):
        pad = cv2.blur(pad, (10, 10))
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

def readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, numImg, flag):
    inputPath = os.path.join(inputFolder, inputImgName)
    inputImg = cv2.imread(inputPath)
    reshapedImg = reShape(inputPath, inputImg, angle, flag)


    normalOutputImg = Image.fromarray(resizeAndPad(inputImg, flag))
    outputImgName = str(numImg) + ".png"
    outputPath = os.path.join(normalOutputFolder, outputImgName)
    print("to: " + outputPath)
    normalOutputImg.save(outputPath)

    rotatedOutputImg = Image.fromarray(reshapedImg)
    outputImgName = str(numImg) + ".png"
    outputPath = os.path.join(rotatedOutputFolder, outputImgName)
    print("and: " + outputPath)
    rotatedOutputImg.save(outputPath)


for i in range(0, 110):
    index = i
    angle = MAXANGLE * random.random()

    rotations['Train_Rotations'].append(angle)

    inputFolder = INPUT_IMAGES_FOLDER
    normalOutputFolder = "Train_Images"
    rotatedOutputFolder = "Train_Images_Unaligned"
    inputImgName = "mask" + zeroPadName(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder,  rotatedOutputFolder, inputImgName, angle, index, False) #知道顺序然后顺着读

    inputFolder = INPUT_MASKS_FOLDER
    normalOutputFolder = "Train_Masks"
    rotatedOutputFolder = "Train_Masks_Unaligned"
    inputImgName = str(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index, True)

for i in range(110, 165):
    index = i % 110
    angle = MAXANGLE * random.random()

    rotations['Test_Rotations'].append(angle)

    inputFolder = INPUT_IMAGES_FOLDER
    normalOutputFolder = "Test_Images"
    rotatedOutputFolder = "Test_Images_Unaligned"
    inputImgName = "mask" + zeroPadName(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index, False)

    inputFolder = INPUT_MASKS_FOLDER
    normalOutputFolder = "Test_Masks"
    rotatedOutputFolder = "Test_Masks_Unaligned"
    inputImgName = str(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index, True)


pickle.dump(rotations, open('rotations.dat', "wb"))

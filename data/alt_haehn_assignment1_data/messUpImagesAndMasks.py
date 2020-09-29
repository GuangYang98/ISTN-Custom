import cv2
import scipy.ndimage as nd
import random

import numpy as np
from PIL import Image
import pickle
import os

#os.mkdir("Test_Images")
#os.mkdir("Test_Masks")
#os.mkdir("Train_Images")
#os.mkdir("Train_Masks")

#os.mkdir("Test_Images_Star")
#os.mkdir("Test_Masks_Star")
#os.mkdir("Train_Images_Star")
#os.mkdir("Train_Masks_Star")

SUBDIR = "istn/data/lucci_128_rotated"

rotations = {}
rotations['Test_Rotations'] = []
rotations['Train_Rotations'] = []


IMAGESHAPE = (88,88)
PADSHAPE = ((20,20), (20,20), (0,0))
MAXANGLE = 60

INPUT_IMAGES_FOLDER =  "Train_In"
INPUT_MASKS_FOLDER = "Train_Out"

def constructAffineMatrices(deg_angle):

    M = np.zeros((2, 3))
    M[0][0] = np.cos(np.radians(deg_angle))
    M[0][1] = -np.sin(np.radians(deg_angle))
    M[1][0] = np.sin(np.radians(deg_angle))
    M[1][1] = np.cos(np.radians(deg_angle))

    return M

def reShape(name, img, angle):
    print("reshaped: " + name)
    res = cv2.resize(img, dsize=IMAGESHAPE, interpolation=cv2.INTER_AREA)
    pad = np.copy(np.pad(res, PADSHAPE, mode='constant', constant_values=0))
    rot = nd.rotate(pad, angle, reshape=False)
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

def readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, numImg):
    inputPath = os.path.join(inputFolder, inputImgName)
    inputImg = cv2.imread(inputPath)
    reshapedImg = reShape(inputPath, inputImg, angle)

    normalOutputImg = Image.fromarray(resizeAndPad(inputImg))
    outputImgName = str(numImg) + ".png"
    outputPath = os.path.join(normalOutputFolder, outputImgName) 
    print("to: " + outputPath)
    normalOutputImg.save(outputPath)

    rotatedOutputImg = Image.fromarray(reshapedImg)
    outputImgName = str(numImg) + ".png"
    outputPath = os.path.join(rotatedOutputFolder, outputImgName) 
    print("and: " + outputPath)
    rotatedOutputImg.save(outputPath)

for i in range(0, 101):
    index = i
    angle = MAXANGLE * random.random()

    inputFolder = INPUT_IMAGES_FOLDER
    normalOutputFolder = "Train_Images"
    rotatedOutputFolder = "Train_Images_Star"
    inputImgName = "mask" + zeroPadName(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index)

    inputFolder = INPUT_MASKS_FOLDER
    normalOutputFolder = "Train_Masks"
    rotatedOutputFolder = "Train_Masks_Star"
    inputImgName = str(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index)

affineMatrices = []
for i in range(101, 161):
    index = i % 100 - 1
    angle = MAXANGLE * random.random()

    affineMatrices.append(constructAffineMatrices(angle))

    inputFolder = INPUT_IMAGES_FOLDER
    normalOutputFolder = "Test_Images"
    rotatedOutputFolder = "Test_Images_Star"
    inputImgName = "mask" + zeroPadName(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index)

    inputFolder = INPUT_MASKS_FOLDER
    normalOutputFolder = "Test_Masks"
    rotatedOutputFolder = "Test_Masks_Star"
    inputImgName = str(i) + ".png"
    readReshapeOutputImage(inputFolder, normalOutputFolder, rotatedOutputFolder, inputImgName, angle, index)

def constructAffineMatrices(deg_angle):

    M = np.zeros((2, 3))
    M[0][0] = np.cos(np.radians(deg_angle))
    M[0][1] = -np.sin(np.radians(deg_angle))
    M[1][0] = np.sin(np.radians(deg_angle))
    M[1][1] = np.cos(np.radians(deg_angle))

    return M

if (len(affineMatrices) != 60):
    raise Exception("wtf happened")


pickle.dump(affineMatrices, open('originalAffineMatrices.dat', "wb"))

import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from pymira.img.transforms import Resampler
from pymira_custom.nets.stn import STN2D, BSplineSTN2D, STN3D, BSplineSTN3D
from pymira.img.datasets import ImageSegRegDataset
# from __future__ import print_function
import cv2
import pandas as pd


MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15

def alignImages(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg, h


if __name__ == '__main__':

    testPath = "./data/Mess_4040/val.csv"
    img_data = pd.read_csv(testPath)

    i = 1
    hList = []

    for idx in range(len(img_data)):
        src_path = img_data.iloc[idx, 0]
        print("Reading reference image : ", src_path)
        imReference = cv2.imread(src_path, cv2.IMREAD_COLOR)

        trg_path = img_data.iloc[idx, 1]
        print("Reading image to align : ", trg_path);

        imReference = cv2.imread(trg_path, cv2.IMREAD_COLOR)
        im = cv2.imread(src_path, cv2.IMREAD_COLOR)

        source = sitk.ReadImage(src_path, sitk.sitkFloat32)
        target = sitk.ReadImage(trg_path, sitk.sitkFloat32)

        print("Aligning images ...")
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        imReg, h = alignImages(im, imReference)
        #matrix = np.zeros([1, 2, 3], dtype=np.float32)
        #matrix[:, :, :] = h[0:2, :]
        hList.append([h])


        # Write aligned image to disk.
        outFilename = str(idx+1)+".jpg"
        print("Saving aligned image : ", outFilename);
        cv2.imwrite(outFilename, imReg)

        # Print estimated homography
        # print("Estimated homography : \n", h)

    pickle.dump(hList, open("h.dat", "wb"))


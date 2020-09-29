import os
import csv
import itertools
import numpy as np
import cv2
from io import StringIO

# change these to calculate the variance for a different folder/directory
dirRoot = "/home/yahiya.hussain001/istn/haehnassignment1_align"
dirFolders = ['haehnassignment1_e', 'haehnassignment1_i', 'haehnassignment1_u', 'haehnassignment1_s']
dirSubFolders = ['warpedSource', 'refinedWarpedSource']
valCsvPath = "/home/yahiya.hussain001/istn/data/haehn_assignment1_data/val.csv"
nImages = 60

# UNIT TESTS
for d in itertools.product(dirFolders, dirSubFolders):

    path = os.path.join(dirRoot, d[0], d[1])
    if (not os.path.isdir(path)):
        raise Exception("Not found: " + path)

    nItems = len(os.listdir(path))
    if (nItems != nImages):
        print(os.listdir(path))
        raise Exception("Should be " + str(nImages) + " images but in found: " + path + ": " + str(nItems))

if (not os.path.exists(valCsvPath)):
    raise Exception("where is the val.csv?")

if (np.mean(np.array([1,2,3,4])) != 2.5):
    raise Exception("mean doesn't work")
if (np.var(np.array([1,2,3,4])) != 1.25):
    raise Exception("variance doesn't work")



with open("trash.txt", "w") as f:
    f.write("a")
    f.write("b")

with open("trash.txt", "r") as f: 
    if (f.read() != "ab"):
        raise Exception("read write doesn't work")
os.remove("trash.txt")
with StringIO() as out:
    out.write("text")
    if (out.getvalue() != 'text'):
        print(out.getvalue())
        raise Exception("StringIO doesn't work")

#CODE
mseResults = {}
sourceImagePaths = []
targetImagePaths = []
lineCount = 0

# get val.csv source,target contents into memory
with open(valCsvPath) as csv_file:
    
    csv_reader = csv.reader(csv_file, delimiter=',' )

    for row in csv_reader:
        if (row[0] == 'src'):
            continue
        sourceImagePaths.append(row[0])
        targetImagePaths.append(row[1])
        lineCount = lineCount + 1

print("val.csv has number of lines: " + str(lineCount))

if (lineCount != nImages):
    raise Exception("wrong number of lines in csv, should be" + str(nImages))


# calculate the mse of each image compared against the target image in val.csv and average all images and store
results = []
for d in itertools.product(dirFolders, dirSubFolders):

    path = os.path.join(dirRoot, d[0], d[1])
    imageNames = os.listdir(path)

    mseArray = []
    for index, iN in enumerate(imageNames):
        A = cv2.imread(os.path.join(path, iN))
        B = cv2.imread(targetImagePaths[index])

        mse = (np.square(A - B)).mean(axis=None)
        mseArray.append(mse)

    if (len(mseArray) != nImages):
        raise Exception("wrong number of mses")

    mseAverage = np.mean(mseArray)
    mseVariance = np.var(mseArray)

    results.append((path, str(mseAverage), str(mseVariance)))

# calculate the mse average and variance of source vs target in val.csv as the control
mseArray = []
for index, iN in enumerate(imageNames):
    A = cv2.imread(sourceImagePaths[index])
    B = cv2.imread(targetImagePaths[index])

    mse = (np.square(A - B)).mean(axis=None)
    mseArray.append(mse)

if (len(mseArray) != nImages):
    raise Exception("wrong number of mses")

mseAverage = np.mean(mseArray)
mseVariance = np.var(mseArray)
results.append(("original val.csv", str(mseAverage), str(mseVariance)))


# store the results in results.txt
with open("results.txt", "w") as f:
    for r in results:
        with StringIO() as out:
            out.write("Path: ")
            out.write(r[0] + "\n")
            out.write("MseAverage: ")
            out.write(r[1] + "\n")
            out.write("MseVariance: ")
            out.write(r[2] + "\n")
            out.write("\n")
            f.write(out.getvalue())
        


    




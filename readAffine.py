import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import cv2
from pymira.img.transforms import Resampler
from pymira_custom.nets.stn import STN2D, BSplineSTN2D, STN3D, BSplineSTN3D
from pymira.img.datasets import ImageSegRegDataset

name = 'Mess6060_i'
dataname = 'Mess_6060'

BASELINE = False
if (BASELINE):
    df = open('./h.dat', 'rb')
else:
    df = open('./output/'+name+'/affineMatrices_refined_i.dat', 'rb')

source = './data/'+dataname+'/Test_Image_Unaligned/1.png'
out_source = './output/'+name+'/align/'
data = pickle.load(df)
a1 = data[0][0]
theta = torch.tensor(a1)
df.close()

device = torch.device("cpu")

configPath = './data/'+dataname+'/config.json'
with open(configPath) as f:
    config = json.load(f)
resampler_img = Resampler(config['spacing'], config['size'])
resampler_seg = Resampler(config['spacing'], config['size'])

testPath = "./data/"+dataname+"/val.csv"
test_segPath = "./data/"+dataname+"/val.seg.csv"
dataset_test = ImageSegRegDataset(testPath, test_segPath, None, resampler_img=resampler_img, resampler_seg=resampler_seg)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)

cat_matrix = torch.zeros([1, 1, 3], dtype=torch.float)
cat_matrix[:, 0, 2] = 1.0;  #give a value


for i, batch_samples in enumerate(dataloader_test):
    source = batch_samples['source'].to(device)

    if i == 0:
        out_dir = os.path.join(out_source, str(i) + '.png')
        if (BASELINE):
            source = source.cpu().squeeze().numpy().astype(np.uint8)
            height, width = source.shape
            wrp = cv2.warpPerspective(source, theta.numpy(), (width, height))
            plt.imshow(wrp.astype(np.uint8))
            plt.imsave(out_dir, wrp.astype(np.uint8))
        else:
            grid = F.affine_grid(theta, source.size()).to(device)
            wrp = F.grid_sample(source, grid)


            plt.imshow(wrp.cpu().squeeze().numpy().astype(np.uint8))
            plt.imsave(out_dir, wrp.cpu().squeeze().numpy().astype(np.uint8))
        plt.show()


    else:

        ai = data[i][0]
        thetaI = torch.tensor(ai)
        out_dir = os.path.join(out_source, str(i) + '.png')

        if (BASELINE):
            theta = torch.mm(theta, thetaI)
            source = source.cpu().squeeze().numpy()
            height, width = source.shape
            wrp = cv2.warpPerspective(source, theta.numpy(), (width, height))
            plt.imshow(wrp);
            plt.imsave(out_dir, wrp);
        else:
            catThetaI = torch.cat((thetaI, cat_matrix), 1)

            catTheta = torch.cat((theta, cat_matrix), 1)

            cat_mm = torch.bmm(catTheta, catThetaI)

            theta = cat_mm[:, 0:2, :]


            grid = F.affine_grid(theta, source.size()).to(device)
            wrp = F.grid_sample(source, grid)


            plt.imshow(wrp.cpu().squeeze().numpy().astype(np.uint8))
            plt.imsave(out_dir, wrp.cpu().squeeze().numpy().astype(np.uint8))
        plt.show()

    #break
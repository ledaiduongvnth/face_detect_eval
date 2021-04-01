from __future__ import print_function
import numpy as np
import os
import cv2
import torch
from scipy.io import savemat
from scipy.io import loadmat
from t_centerface.centerface import CenterFace

# Folder for saving face detection results
save_folder = "./z/"
# Folder for containing input images
dataset_folder = "./widerface_val_TEST/images/"


def main():

    with open("./widerface_val_TEST/1.txt", 'w') as f:
        for folder in os.listdir(dataset_folder):
            print("-----folder--------: ", folder)
            for file in os.listdir(dataset_folder + folder):
                print("-----file--------: ", file)
                f.write("/" + folder + "/" + file + "\n")

main()
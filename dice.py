import pdb
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math

PATH = '../img/images/'

#Set of accepted image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','jpx','j2k','j2c','fpx', \
                  'pcd','pdf','png','ppm','webp','bmp','bpg','dib','wav', \
                  'cgm','svg','gif'])


def init_kernel(values):
    '''
    Initialize a kernel with the given values
    @params:
        values: array of values to be used to establish the kernel
    @return:
        kernel: kernel with dimensions (M x M) where M = length of values
    '''
    kernel = np.empty([len(values),len(values)])
    ki = values
    kj = values

    for i in range(len(values)):
        for j in range(len(values)):
            kernel[i][j] = ki[i] * kj[j]

    return kernel


def morph():
    #TODO implement algorithm
    return None


def main():
    if (len(sys.arv) < 2):
        print('Need target directory')
        sys.exit()
    path = sys.argv[1]

    if not os.path.isdir(path) and path.split('.')[1].lower() in EXTENSIONS:
        global PATH
        PATH = path
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        kernel = init_kernel([.25,.5,.25])
        blur = cv.GaussianBlur(img,(5,5),0)
        ret,th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

        plt.imshow(ret,'ret')
        plt.imshow(th,'th')
        plt.show()

        #TODO Implement algorithm

        #threshold( gray, thr, 100,255,THRESH_BINARY ); Greyscale -> binary

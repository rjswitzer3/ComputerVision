'''
@Author Ryan Switzer

Personal implemnetation of the canny edge detection algorithm using OpenCV.
'''
import cv2
import numpy as np
import sys
import os
import math
from scipy.signal import convolve2d


def ritconv(rows,cols,kr,kc,img,out):
    '''
    Basic 2D convolution
    @params:
        rows: number of horizontal pixels
        cols: number of vertical pixels
        kr: kernal row dimension
        kc: kernel column dimension
    @return:

    '''
    # TODO consider faster algorithm

    kCenterX = kc/2
    kCenterY = kr/2

    for i in range(rows):
        for j in range(cols):
            #Kernel
            for ki in range(kr):
                fki = kRows - 1 - ki
                for kj in range(kc):
                    fkj = kCols - 1 - kj

                    sig_i = i + (ki-kCenterY)
                    sig_j = j + (kj+kCenterX)

                    if (sig_i >= 0) and (sig_i < rows) \
                        (sig_j >= 0) and (sig_j < cols):
                        out[i][j] += img[sig_i][sig_j] * kernel[ki][kj]


def ritgrad():
    # TODO Implement me
    return None


def ritcan():
    # TODO Implement me
    return None


def main():
    if (len(sys.argv) < 2):
        print('Need target file') # TODO Do we need to process multiple imgs?
        sys.exit()
    img = sys.argv[1]

main()

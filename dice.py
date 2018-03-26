import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import math
from scipy.signal import convolve2d

PATH = 'images/'

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


def write_result(imgs,img,name):
    '''
    Write the results of the canny edge detection
    @params:
        imgs: list of all images
        img: the resultant image post manipulation
        name: the name descriptor for the image
    @returns:
        None
    '''
    newfile = '.'.join(PATH.split('.')[:-1]) + '_'+name+'.' + PATH.split('.')[-1]

    if imgs != None:
        # Create side-by-side comparison and write
        result = np.hstack(imgs)
        cv.imwrite(newfile,result)
    else:
        cv.imwrite(newfile, img)


def init(img):
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobelx
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #sobely

    gx = convolve2d(img,sobelx)
    gy = convolve2d(img,sobely)

    gradient = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
    theta = np.arctan2(gy,gx)
    thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5

    return [gradient, thetaQ]


def main():
    '''
    Main function handling input, output and the program's operation flow
    @params:
        None
    @returns:
        None
    '''
    if (len(sys.argv) < 2):
        print('Need target image')
        sys.exit()
    path = sys.argv[1]

    if not os.path.isdir(path) and path.split('.')[1].lower() in EXTENSIONS:
        global PATH
        PATH = path

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        gb_img = cv.GaussianBlur(img,(5,5),0)
        grad,theta = init(gb_img)

        #thresh = cv.threshold(gb_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        thresh = cv.adaptiveThreshold(gb_img, 255, \
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    cv.THRESH_BINARY_INV,11,1)
        kernel = np.ones((3,3),np.uint8)
        erode = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=3)

        write_result([img,gb_img,thresh,erode],None,'progression')

        #plt.imshow(ret,'ret')
        #plt.imshow(th,'th')
        #plt.show()

        #threshold( gray, thr, 100,255,THRESH_BINARY ); Greyscale -> binary


main()

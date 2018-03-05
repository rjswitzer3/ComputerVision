'''
@Author Ryan Switzer

Rudimentary implementation of the canny edge detection algorithm written in
python using OpenCV. Resulting images of the convolution shall be written to
disk using the following naming convention <image name>_conv.<extension>.
'''
import cv2
import numpy as np
import sys
import os
import math
from scipy.signal import convolve2d

#Set of accepted image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','jpx','j2k','j2c','fpx', \
                  'pcd','pdf','png','ppm','webp','bmp','bpg','dib','wav', \
                  'cgm','svg','gif'])


def init_kernel(values):
    '''
    Initialize a kernel with the given size
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


def test_conv(img,conv,kernel,path):
    '''
    Quick test function to write the results of convolution side-by-side
    @params:
        img: the original input images
        conv: the resultant image post convolution
        kernel: the filter used for the convolution
        path: the image file path
    @returns:
        None
    '''
    test_2d = convolve2d(img,kernel)
    comp = np.hstack((conv,test_2d))

    newfile = '.'.join(path.split('.')[:-1]) + '_TEST-CONV.' + path.split('.')[-1]
    cv2.imwrite(newfile, comp)


def write_result(img,conv,kernel,path):
    '''
    Write the results of the canny edge detection
    @params:
        img: the original input images
        conv: the resultant image post convolution
        kernel: the filter used for the convolution
        path: the image file path
    @returns:
        None
    '''
    newfile = '.'.join(path.split('.')[:-1]) + '_conv.' + path.split('.')[-1]

    # Create side-by-side comparison and write
    result = np.hstack((img,conv))
    cv2.imwrite(newfile, result)

    #Test that the convolution implementation is working correctly
    #test_conv(img,conv,kernel,path)


def ritconv(img,kernel):
    '''
    Preforms basic 2D convolution on the input image. Convolution uses float32
    @params:
        img: the input image to preform the convolution on
        kernel: the filter used for the convolution
    @returns:
        conv_img: the image after convolution
    '''
    #Get the height, width and # of channels for the image
    height,width = img.shape
    #Compute the center position of the kernel (assumes square kernel)
    kCenterX = int(round(len(kernel)/2.0))
    kCenterY = int(round(len(kernel)/2.0))
    k = len(kernel)
    conv_img = np.empty([width,height])

    # Traverse image
    for i in range(width):
        for j in range(height):
            # Traverse kernel
            for ki in range(k):
                #Flipped kernel row index
                fki = k - 1 - ki
                for kj in range(k):
                    #Flipped kernel column index
                    fkj = k - 1 - kj
                    # Input signal index; for boundary check
                    si = i + (ki-kCenterX)
                    sj = j + (kj-kCenterY)
                    # Keep convolution within image boundaries
                    if (si >= 0) and (si < width) and \
                        (sj >= 0) and (sj < height):
                        conv_img[i][j] += img[si][sj].astype(np.float32) * kernel[fki][fkj]

    return conv_img


def ritgrad():
    # TODO Implement me
    return None


def ritcan(image,scale,weak,strong):
    # TODO Implement me
    return None


def main():
    '''
    Main function handling input, output and the program's operational flow
    @params:
        none
    @returns:
        none
    '''
    if (len(sys.argv) < 2):
        print('Need target file')
        sys.exit()
    path = sys.argv[1]

    #Various filters
    #kernel = init_kernel([1,4,6,4,1])
    #kernel = init_kernel([.00390625,.015625,.0234375,.015625,.00390625])
    #kernel = init_kernel([.25,.5,.25])
    #kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobelx
    #kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #sobely
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) #Laplacian


    if not os.path.isdir(path) and path.split('.')[1].lower() in EXTENSIONS:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        conv_img = ritconv(img,kernel)

        write_result(img,conv_img,kernel,path)
    else:
        print('File format not supported and/or file does not exist')


main()

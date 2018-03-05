'''
@Author Ryan Switzer

Rudimentary implementation of the canny edge detection algorithm written in
python using OpenCV. Resulting images of the algorithm shall be written to
disk using the following naming convention <image name>_<conv|grad|edge>.<extension>.
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
#Path where the image shall exist
PATH = 'TEST_IMAGES/'


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


def test_conv(img,conv,kernel):
    '''
    Quick test to verify results of ritconv() and convolve2d are equal
    @params:
        img: the original input images
        conv: the resultant image post convolution
        kernel: the filter used for the convolution
    @returns:
        None
    '''
    test_2d = convolve2d(img,kernel)

    #TODO Investigate how to correct test
    #Result of convole2d is bigger than original, thus proceeding check fails
    print(np.array_equal(conv,test_2d))

    #Write results to preform eye test
    #comp = np.hstack((conv,test_2d))
    #newfile = '.'.join(PATH.split('.')[:-1]) + '_TEST-CONV.' + PATH.split('.')[-1]
    #cv2.imwrite(newfile, test_2d)


def write_result(img,conv,kernel,name):
    '''
    Write the results of the canny edge detection
    @params:
        img: the original input images
        conv: the resultant image post convolution
        kernel: the filter used for the convolution
        name: the name descriptor for the image
    @returns:
        None
    '''
    newfile = '.'.join(PATH.split('.')[:-1]) + '_'+name+'.' + PATH.split('.')[-1]

    # Create side-by-side comparison and write
    #result = np.hstack((img,conv))
    cv2.imwrite(newfile, conv)

    #Test that the convolution implementation is working correctly
    test_conv(img,conv,kernel)


def nms():
    # TODO Implement me
	return None


def hysteresis():
    # TODO Implement me
    return None


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


def ritgrad(conv):
    '''
    @params:
        conv: image with noise reduction
    @returns:
        gradient: matrix containing the gradient magnitudes as floats
        angles: matrix containing the gradient angle
    '''
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobelx
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #sobely

    gx = ritconv(conv,sobelx)
    gy = ritconv(conv,sobely)

    gradient = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
    angles = np.arctan2(gy,gx)

    return [gradient, angles]


def ritcan(image,scale,weak,strong):
    '''
    Implementation of the canny edge detection algorithm
    @params:
        image: the input image to preform canny edge detection on
        scale: the kernel used for the convolution (i.e. Gaussian filter)
        weak: the weak edge threshold
        strong: the strong edge threshold
    @returns:
        none
    '''
    kernel = scale
    conv_img = ritconv(image,kernel)
    gradient,angles = ritgrad(conv_img)

    write_result(image,conv_img,kernel,'conv')
    write_result(image,gradient,kernel,'grad')
    #TODO implement non-maximum-supression
    #TODO implement hysteresis threshold


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

    if not os.path.isdir(path) and path.split('.')[1].lower() in EXTENSIONS:
        global PATH
        PATH = path
        kernel = init_kernel([.25,.5,.25]) #Gaussian
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        ritcan(img,kernel,100,200)
    else:
        print('File format not supported and/or file does not exist')


main()

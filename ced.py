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


def nms(img,grad,thetaQ):
    '''
    Preforms non-maxmimum supression on the gradient matrix to derive the
    @params:

    @returns:

    '''
    grad_sup = grad.copy()
    height,width = img.shape

    for i in range(width):
        for j in range(height):
            if (i == 0) or (i == width-1) or (j == 0) or (j == height - 1):
                grad_sup[i][j] = 0
                continue

            tq = thetaQ[i][j] % 4
            if tq == 0:
                if (grad[i][j] <= grad[i][j-1]) or (grad[i][j] <= grad[i][j+1]):
                    grad_sup[i][j] = 0
            if tq == 1:
                if (grad[i][j] <= grad[i-1][j+1]) or (grad[i][j] <= grad[i+1][j-1]):
                    grad_sup[i][j] = 0
            if tq == 2:
                if (grad[i][j] <= grad[i-1][j]) or (grad[i][j] <= grad[i+1][j]):
                    grad_sup[i][j] = 0
            if tq == 3:
                if (grad[i][j] <= grad[i-1][j-1]) or (grad[i][j] <= grad[i+1][j+1]):
                    grad_sup[i][j] = 0

    return grad_sup


def hysteresis(img,grad_sup,strong,weak):
    '''
    Determines which edges are really edges and whicha are not using
    thresholding and tracing the edges via finding weak edge pixels near strong
    edge pixels.
    @params:
        img: the original images
        grad_sup: the gradient supression resultant matrix
        strong: upper threshold value
        weak: lower threshold value
    @returns:
        final_edges: extened strong edges
    '''
    strong_edges = (grad_sup > strong)
    threshold_edges = np.array(strong_edges, dtype=np.uint8) + (grad_sup > weak)
    final_edges = strong_edges.copy().astype(np.float32)
    height,width = img.shape
    pixels = []

    for i in range(1, width-1):
        for j in range(1, height-1):
            if threshold_edges[i][j] != 1:
                continue
            local = threshold_edges[i-1:i+2,j-1:j+2]
            local_max = 0
            try:
                local_max = local.max()
            except ValueError:
                pass
            if local_max == 2:
                pixels.append((i,j))
                final_edges[i][j] = 1

    while len(pixels) > 0:
        newPixels = []
        for i,j in pixels:
            for di in range(-1,2):
                for dj in range(-1,2):
                    if di == 0 and dj == 0:
                        continue
                    ii = i+di
                    jj = j+dj
                    if threshold_edges[ii][jj] == 1 and final_edges[ii][jj] == 0:
                        newPixels.append((ii,jj))
                        final_edges[ii][jj] = 1
        pixels = newPixels

    return final_edges


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
        theta: matrix containing the gradient angles
    '''
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobelx
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #sobely

    gx = ritconv(conv,sobelx)
    gy = ritconv(conv,sobely)

    gradient = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
    theta = np.arctan2(gy,gx)

    return [gradient, theta]


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
    grad,theta = ritgrad(conv_img)

    thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5
    grad_sup = nms(image,grad,thetaQ)
    edges = hysteresis(image,grad_sup,strong,weak)
    print(type(edges[0][0]))
    print(type(grad_sup[0][0]))

    write_result(image,conv_img,kernel,'conv')
    write_result(image,grad,kernel,'grad')
    #cv2.imshow(edges)
    write_result(image,grad_sup,kernel,'sup')
    write_result(image,edges,kernel,'edge')


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

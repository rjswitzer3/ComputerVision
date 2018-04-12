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


def test_conv(img,kernel):
    '''
    Quick test to verify results of ritconv() against convolve2d()
    (i.e. confirm margin of error within 1%)
    @params:
        img: the original input images
        kernel: the filter/kernel used for the convolution
    @returns:
        None
    '''
    conv = ritconv(img,kernel)
    test_2d = convolve2d(img,kernel)

    print('Margin of error (%) of ritconv() vs convolve2d()')
    print(np.mean(conv != test_2d).sum()/float(conv.size))


def test_can(img,edges):
    '''
    Quick test to verify the results of ritcan() against cv2.canny()
    (i.e. confirm margin of error within 1% )
    @params:
        img: the original input images
        edges: the result of ritcan()
    @returns:
        None
    '''
    can = cv2.Canny(img,100,200)

    print('Margin of error (%) of ritcan() vs cv2.canny()')
    print(np.mean(edges != can).sum()/float(edges.size))


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
        cv2.imwrite(newfile,result)
    else:
        cv2.imwrite(newfile, img)


def nms(img,grad,thetaQ):
    '''
    Preforms non-maxmimum supression on the gradient image to derive an image
    with thin edges
    @params:
        img: the oringal input images
        grad: the gradient resultant images
        thetaQ: the quantized gradient direction
    @returns:
        grad_sup: the non-maximum suppression resultant image
    '''
    grad_sup = grad.copy()
    height,width = img.shape

    #Non-maximum suppression
    for i in range(width):
        for j in range(height):
            #Suppress pixels on image edge
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
    Determines which edges are really edges and which are not using
    thresholding to trace the edges by finding weak edge pixels near strong
    edge pixels. (i.e. remove small pixel noise)
    @params:
        img: the original images
        grad_sup: the gradient supression resultant matrix
        strong: upper threshold value
        weak: lower threshold value
    @returns:
        final_edges: image with 'strong' edges
    '''
    strong_edges = (grad_sup > strong)
    threshold_edges = np.array(strong_edges, dtype=np.uint8) + (grad_sup > weak)
    final_edges = strong_edges.copy().astype(np.float32)
    height,width = img.shape
    pixels = []

    # Tace edges & Find weak edge pixels near strong edge pixels
    for i in range(1, width-1):
        for j in range(1, height-1):
            if threshold_edges[i][j] != 1:
                continue
            local = threshold_edges[i-1:i+2,j-1:j+2]
            local_max = local.max()
            if local_max == 2:
                pixels.append((i,j))
                final_edges[i][j] = 1

    # Extend strong edges
    while len(pixels) > 0:
        new_pixels = []
        for i,j in pixels:
            for di in range(-1,2):
                for dj in range(-1,2):
                    if di == 0 and dj == 0:
                        continue
                    ii = i+di
                    jj = j+dj
                    if threshold_edges[ii][jj] == 1 and final_edges[ii][jj] == 0:
                        new_pixels.append((ii,jj))
                        final_edges[ii][jj] = 1
        pixels = new_pixels

    # Set edge pixel to maximum intensity
    for i in range(width):
        for j in range(height):
            if final_edges[i][j] == 1:
                final_edges[i][j] = 255

    return final_edges


def ritconv(img,kernel):
    '''
    Preforms rudimentary 2D convolution on the input image.
    Convolution imposes usage of float32.
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

    write_result(None,conv_img,'conv')
    write_result(None,grad,'grad')
    write_result(None,edges,'edge')
    write_result([conv_img,grad,edges],None,'all')

    return edges


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
        edges = ritcan(img,kernel,100,200)

        # TESTS for error margin
        test_conv(img,kernel)
        test_can(img,edges)

    else:
        print('File format not supported and/or file does not exist')


main()

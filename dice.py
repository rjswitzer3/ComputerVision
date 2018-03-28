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

def output_result(dice):
    '''
    Write out to the console the results of the processing for the given image.
    @params:
        dice: dictionary encapsulating all pertient dice information for the image
    @returns:
        None
    '''
    print('INPUT Filename\t\t'+PATH.split('/')[1])
    print('Number of Dice:\t\t'+str(dice['count']))
    print('Number of 1\'s:\t\t'+str(dice[1]))
    print('Number of 2\'s:\t\t'+str(dice[2]))
    print('Number of 3\'s:\t\t'+str(dice[3]))
    print('Number of 4\'s:\t\t'+str(dice[4]))
    print('Number of 5\'s:\t\t'+str(dice[5]))
    print('Number of 6\'s:\t\t'+str(dice[6]))
    print('Number of Unknown:\t'+str(dice['unknown']))
    print('Total of all dots:\t'+str(dice['total_sum']))


def write_result(imgs,img,name):
    '''
    Write the images resulting from transformations and morphology
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


def init(img):
    '''
    Derive pertinent gradient information from the designated image
    @params:
        conv: image with noise reduction
    @returns:
        gradient: matrix containing the gradient magnitudes as floats
        thetaQ: matrix containing the gradient angles (quantized)
    '''
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #sobelx
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) #sobely

    gx = convolve2d(img,sobelx)
    gy = convolve2d(img,sobely)

    gradient = np.sqrt(gx.astype(np.float32)**2 + gy.astype(np.float32)**2)
    theta = np.arctan2(gy,gx)
    thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5

    return [gradient, thetaQ]


def init_detector():
    '''
    Initialize and define a blob detector with custom parameters
    @params
        none
    @returns
        SimpleBlobDetector: SimpleBlobDetector with custom parameters
    '''
    params = cv.SimpleBlobDetector_Params()
    params.filterByInertia = True
    params.minInertiaRatio = .6
    params.filterByConvexity = True
    params.minConvexity = .5
    params.filterByCircularity = True
    params.minCircularity = .2

    v = (cv.__version__).split('.')
    if int(v[0]) < 3:
        return cv.SimpleBlobDetector(params)
    else:
        return cv.SimpleBlobDetector_create(params)


def allocate_dice(dice,keys):
    '''
    Allocate and increment the data in the dice dictionary according
    to the keys provided.
    @params:
        dice: dictionary encapsulating all pertient dice information for the image
        keys: key matches found in the image from the blob detector
    @returns:
        dice: updated dictionary encapsulating all pertient dice information
              for the image
    '''
    if len(keys) > 0 and len(keys) < 7:
        dice[len(keys)] += 1
        dice['count'] += 1
        dice['total_sum'] += len(keys)
    else:
        dice['unknown'] += 1
        print('Pip deciphering error')
    return dice


def decipher_dice(gb_img):
    '''
    Main pipeline for deciphering dice in the image provide.
    @params:
        gb_img: input image with noise reduction
    @returns:
        dice: array of dice images derived from the input image
    '''
    dice = []
    area = 0
    connectivity = 4
    roi = gb_img.copy()
    kernel = np.ones((3,3),np.uint8)

    r,thresh = cv.threshold(gb_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cc = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S) #TODO
    print(cc)
    close = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 23)
    can_img = cv.Canny(close,200,330)

    img,cons,hier = cv.findContours(can_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    for c in cons:
        area = cv.contourArea(c)
        if (area > 2000) or (area < 3500):
            rec = cv.minAreaRect(c)
            box = cv.boxPoints(rec)
            box = np.int0(box)
            roi = cv.drawContours(roi,[box],0,(0,0,255),10)

            rect = cv.boundingRect(c)
            x,y,w,h = rect
            if h>100 and w>100: #Super Hacky TODO Find sophisticated solution
                dice.append(gb_img[y:y+h,x:x+w])

    write_result([thresh,close,can_img,roi],None,'output')

    return dice


def count_pips(dice_imgs):
    '''
    For each di image derived, count the number of pips (i.e. its face value)
    @params:
        dice_imgs: array of dice images derived from the input image
    @returns:
        dice: dictionary encapsulating all pertient dice information for the image
    '''
    dice = { 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, \
            'unknown': 0, 'total_sum': 0, 'count': 0 }
    kernel = np.ones((3,3),np.uint8)
    detector = init_detector()

    i = 0
    for di in dice_imgs:
        r,thresh = cv.threshold(di, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        di_fill = thresh.copy()
        h,w = di_fill.shape[:2]
        mask = np.zeros((h+2,w+2), np.uint8)

        cv.floodFill(di_fill,mask,(0,0),255)
        cv.floodFill(di_fill,mask,(0,h-1),255)
        cv.floodFill(di_fill,mask,(w-1,0),255)
        cv.floodFill(di_fill,mask,(w-1,h-1),255)

        pips = thresh | di_fill
        pips = cv.morphologyEx(pips, cv.MORPH_DILATE, kernel, iterations = 5)
        pips = cv.morphologyEx(pips, cv.MORPH_ERODE, kernel, iterations = 10)

        keys = detector.detect(pips)

        #write_result(None,pips,'di-'+str(i)+'-val'+str(len(blobs)))
        i += 1
        dice = allocate_dice(dice,keys)

    return dice


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

        kernel = init_kernel([.25,.5,.25])

        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        gb_img = cv.GaussianBlur(img,(5,5),0)

        grad,theta = init(gb_img)# TODO remove if unused

        dice_imgs = decipher_dice(gb_img)
        dice = count_pips(dice_imgs)
        output_result(dice)




main()

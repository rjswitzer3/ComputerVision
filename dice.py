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
    print('INPUT Filename\t\t'+PATH.split('/')[1])
    print('Number of Dice:\t\t')
    print('Number of 1\'s:\t\t')


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
    params = cv.SimpleBlobDetector_Params()
    params.filterByInertia = True
    params.minInertiaRatio = .6
    params.filterByConvexity = True
    params.minConvexity = .5

    v = (cv.__version__).split('.')
    if int(v[0]) < 3:
        return cv.SimpleBlobDetector(params)
    else:
        return cv.SimpleBlobDetector_create(params)


def sand(gb_img):
    kernel = np.ones((3,3),np.uint8)

    r,thresh = cv.threshold(gb_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    open = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 8)
    can_img = cv.Canny(open,200,330)


def decipher_dice(gb_img):
    '''
    MAIN PIPELINE FOR COUNTING DICE
    '''
    dice = []
    area = 0
    connectivity = 4
    roi = gb_img.copy()
    kernel = np.ones((3,3),np.uint8)

    r,thresh = cv.threshold(gb_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cc = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S) #TODO
    open = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 23)
    can_img = cv.Canny(open,200,330)

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

    write_result([thresh,open,can_img,roi],None,'output')

    return dice


def count_pips(gb_img,dice_imgs):
    dice = { 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 0, 6 : 0, 'total': 0, 'count': len(dice_imgs) }
    kernel = np.ones((3,3),np.uint8)
    detector = init_detector()
    #r,thresh = cv.threshold(gb_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    i = 0
    for di in dice_imgs:
        r,thresh = cv.threshold(di, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #di_fill = cv.morphologyEx(thresh, cv.MORPH_DILATE, kernel, iterations = 3)

        di_fill = thresh.copy()
        h,w = di_fill.shape[:2]
        mask = np.zeros((h+2,w+2), np.uint8)

        cv.floodFill(di_fill,mask,(0,0),255)
        cv.floodFill(di_fill,mask,(0,h-1),255)
        cv.floodFill(di_fill,mask,(w-1,0),255)
        cv.floodFill(di_fill,mask,(w-1,h-1),255)

        pips = thresh | di_fill
        #pips = cv.morp

        blobs = detector.detect(pips)

        write_result(None,pips,'di-'+str(i)+'-val'+str(len(blobs)))
        i += 1



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

        grad,theta = init(gb_img)

        dice_imgs = decipher_dice(gb_img)
        #dice_vals = count_pips(gb_img,dice_imgs)




main()

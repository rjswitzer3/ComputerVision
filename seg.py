'''
@Author Ryan Switzer

Image segmentation implementation using the k-means algorithm to cluster image
pixels to various color spaces.
'''
import cv2
import numpy as np
from matplotlib import pyplot as plot
import sys
import os


#Set of accepted image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','jpx','j2k','j2c','fpx', \
                  'pcd','pdf','png','ppm','webp','bmp','bpg','dib','wav', \
                  'cgm','svg','gif'])
#Path where the image shall exist
PATH = 'images/'


def euclidean(a, b, ax=1):
    '''
    Calculate the euclidean distance between two points
    '''
    return np.linalg.norm(a - b, axis=ax)


def loss(orig, xform):
    '''
    Compute the sum squared loss between the original and transformed images
    @params:
        orig: the original input image
        xform: the transformed image
    @returns:
        ss: squared sum loss
    '''
    #TODO: Implement
    print('Calculting loss...')


def quantize(image, features, k):
    '''
    Quantize the image into a limited palette colormap
    @params:
        image: the oringal input image
        features: deciphered features in the image
        k: number of iterations to performed
    @returns:
        img: 2D array of cluster IDs with same dimensions as the input image
        cmap: array of cluster IDs corresponding cluster IDs to mean intensities
    '''
    #TODO Implement
    print('Starting quantization...')


def kmeans(samples, k):
    '''
    Performs k-means on array of N-dimensional Samples
    @params:
        samples: image features
        k: number of clusters
    @returns:
        cluster: array of cluster assignment
    '''
    y,x,ch = samples.shape
    cx = np.random.randint(0, np.max(x)-20, size=k)
    cy = np.random.randint(0, np.max(y)-20, size=k)
    cent = np.array(list(zip(cx,cy)))

    cache = np.zeros(cent.shape)
    clusters = np.zeros()
    error = euclidean(c,cache,None)

    print(cx,cy)
    print(cent)
    print(cent[1][0])



def main():
    '''
    Main function handling input, output and the program's operational flow
    @params:
        none
    @returns:
        None
    '''
    if (len(sys.argv) < 2):
        print('Need target image')
        sys.exit()
    path = sys.argv[1]

    if not os.path.isdir(path) and path.split('.')[1].lower() in EXTENSIONS:
        global PATH
        PATH = PATH

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        kmeans(img, 3)
    else:
        print('Error unsupported input - ' + str(path))
        sys.exit()


main()

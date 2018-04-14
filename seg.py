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
#Set of colorspace features
FEATURES = {'HSV' : [0,255], 'GRAY' : [0,255]}
#Threshold for tight cluster centroid
THRESHOLD = 1.0

class Centroid(object):
    def __init__(self,value,id):
        self.value = value
        self.id = id
    def str(self):
        return str(self.id)+" - "+str(self.value)
    def get_id(self):
        return self.id
    def get_value(self):
        return self.value
    def update(self,value):
        self.value = value

class Cluster(object):
    def __init__(self,values,mean,centroid):
        self.values = values
        self.mean = mean
        self.centroid = centroid
    def append(self,value):
        self.values.append(value)
    def get_center(self):
        return self.centroid
    def get_values(self):
        return self.values
    def get_mean(self):
        return np.mean(self.values)



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


def calc_error(centroids, cache):
    '''
    Calculate the average step distance from the prior centroid to the current.
    '''
    t_error = 0.0
    for i in range(len(centroids)):
        t_error += euclidean(centroids[i].get_value(),cache[i],None)
    err = t_error/len(centroids)
    print(err)
    return err


def label(pixel, centroids):
    '''
    Determine the label for the pixel by calculating the minimum difference to
    each centroidself.
    @params:
        pixel: the pixel value at some position on a sample
        centroids: list of all centroids
    @returns:
        id: the corresponding nearest centroid label
    '''

    distances = np.zeros(len(centroids))
    for i in range(len(centroids)):
        distances[i] = euclidean(pixel,centroids[i].get_value())
    min = np.min(distances)
    for c in centroids:
        if min == euclidean(pixel, c.get_value()):
            return c.get_id()


def euclidean(a, b, ax=1):
    '''
    Calculate the euclidean distance between two points
    '''
    return np.linalg.norm(a - b)


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
        k: number of clusters
    @returns:
        img: 2D array of cluster IDs with same dimensions as the input image
        cmap: array of cluster IDs corresponding cluster IDs to mean intensities
    '''
    samples = image.flatten() #TODO - Float32
    #s = image.reshape((image.shape[0]*image.shape[1], 1))

    clusters = kmeans(samples,k)



def kmeans(samples, k):
    '''
    Performs k-means on array of N-dimensional Samples
    @params:
        samples: image features
        k: number of clusters
    @returns:
        cluster: array of cluster assignment
    '''
    s = samples
    labels = [None]*len(s)
    centroids, clusters, cache = ([] for i in range(3)) # values change based on colorsapce

    for i in range(k):
        c = Centroid(np.random.randint(0,255),i)
        centroids.append(c)
        clusters.append(Cluster([],0.0,c))
        cache.append(0.0)
        print(c.str())

    error = calc_error(centroids,cache)

    while error > THRESHOLD:
        for i in range(len(s)):
            labels[i] = label(s[i],centroids)
            clusters[labels[i]].append(s[i])
        #for i in range(k):
        #    cache[i] = centroids[i].get_value()
        cache = [centroids[i].get_value() for i in range(k)]
        for i in range(k):
            centroids[i].update(clusters[i].get_mean())
            clusters[i] = Cluster([],0.0,centroids[i])
        error = calc_error(centroids,cache)

    return labels


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
        #fv = img.reshape((img.shape[0] * img.shape[1],3))
        quantize(img, '', 3)

    else:
        print('Error unsupported input - ' + str(path))
        sys.exit()


main()

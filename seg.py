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
FEATURES = {'GRY' : [[0,255]], \
            'RGB' : [[0,255],[0,255],[0,255]], \
            'HSV' : [[0,180],[0,255],[0,255]], \
            'LAB' : [[0,150],[-128,127],[-128,127]]}
#Threshold for acceptably tight cluster centroid
THRESHOLD = 1.0
#List of all the centroid objects
CENTROIDS = []
################################################################
#Parameter flags
################################################################
K = 6                      #k-value/clusters
F = 'HSV'                   #Feature usage


class Centroid(object):
    '''
    Object representaion of a centroid. Encapsulates the current value and
    the label identifier correlating to a cluster
    '''
    def __init__(self,value,id):
        self.value = value
        self.id = id
    def str(self):
        return "Centroid:"+str(self.id)+" - "+str(self.value)
    def get_id(self):
        return self.id
    def get_value(self):
        return self.value
    def update(self,value):
        self.value = value


class Cluster(object):
    '''
    Object representation of a cluster. Encapsulates all value points relative to
    a centroid, the corresponding centroid and the mean value of the cluster.
    '''
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
        if len(self.values) > 0: #Extreme edge case
            if len(self.values[0]) > 1 and isinstance(self.values[0], np.ndarray):
                return np.array([np.mean(zip(*self.values)[0]),\
                                np.mean(zip(*self.values)[1]),\
                                np.mean(zip(*self.values)[2])])
            else:
                return np.mean(self.values)
        else:
            return 0.0


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
        cv2.imwrite(newfile, result)
    else:
        cv2.imwrite(newfile, img)


def calc_error(cache):
    '''
    Calculate the average step distance from the prior centroid to the current.
    @params:
        cache: list of the prior centroid values
    @returns:
        error: the difference between the cluster average and the centroid
    '''
    error = 0.0
    for i in range(K):
        error += euclidean(CENTROIDS[i].get_value(), cache[i], None)
    error = error/K
    print("Cluster error: "+str(error))                 #TESTING TODO REMOVE
    return error


def euclidean(a, b, ax=1):
    '''
    Wrapper for np.linalg.norm() that calculates the euclidean distance between
    two points
    @params:
        a: value one for calculating euclidean distance
        b: value two for calculating euclidean distance
    @returns:
        euclidean_distance: the euclidean distance between the two values
    '''
    return np.linalg.norm(a - b)


def rand_vectors():
    '''
    Generate random values based on the feature colorspace.
    @params:
        none
    @returns:
        rv: random pixel intensity values
    '''
    rv = np.empty(shape=0)
    for f in FEATURES[F]:
        rv = np.append(rv, np.random.randint(f[0],f[1]))
    return rv


def img_reshape(arr, feature, image, flat):
    '''
    Wrapper for np.reshape() that reshapes the array to the designated depth and
    image dimensions.
    @params:
        arr: the array to Reshape
        feature: string depicting the feature colorspace used for depth
        image: the image to reshape to
        flat: boolean indicating 1D or 2D
    @returns:
        arr: the array reshaped according to the parameters
    '''
    if not flat:
        return arr.reshape((image.shape[0], image.shape[1], len(FEATURES[feature])))
    else:
        return arr.reshape((image.shape[0]*image.shape[1], len(FEATURES[feature])))


def cs_convert(image, features, revert):
    '''
    Wrapper for cv2.cvtColor() that converts the image to the designated feature
    colorspace or reverts from back to RGB.
    @params:
        image: the image to Convert
        features: the colorspace to perform conversions to/from
        revert: boolean flag dictating to/from conversion
    @returns
        image: the image converted to the according feature colorspace
    '''
    if features == 'GRY':
        if revert:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif features == 'HSV':
        if revert:
            return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif features == 'LAB':
        if revert:
            return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        else:
            return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        return image


def label(pixel):
    '''
    Determine the label for the pixel by calculating the minimum difference to
    each centroid.
    @params:
        pixel: the pixel value at some position on a sample
    @returns:
        id: the corresponding nearest centroid label
    '''
    distances = np.zeros(K)
    for i in range(K):
        #Calculate euclidean distance from the pixel to each centroid
        distances[i] = euclidean(pixel,CENTROIDS[i].get_value())
    min = np.min(distances)
    for i in range(K):
        if min == distances[i]:
            return CENTROIDS[i].get_id()


def loss(orig, xform):
    '''
    Compute the sum squared loss between the original and transformed images
    @params:
        orig: the original input image
        xform: the transformed image
    @returns:
        ssl: pixel wise squared sum loss
    '''
    return ((orig-xform)**2).sum()/len(orig.flatten())


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
    #Convert the image based on features
    image = cs_convert(image, features, False)
    #Flatten the image & perform kmeans
    samples = img_reshape(image, features, image, True)
    cluster = kmeans(samples,k)
    cmap = np.zeros(samples.shape)

    #Assign centroid intensity value to each pixel in it's cluster
    for i in range(len(samples)):
        cmap[i] = CENTROIDS[cluster[i]].get_value()

    img = cluster.reshape((image.shape[0], image.shape[1], 1))

    return [img, cmap]


def kmeans(samples, k):
    '''
    Performs k-means on array of N-dimensional Samples
    @params:
        samples: image features
        k: number of clusters
    @returns:
        labels: array of cluster assignment
    '''
    s = samples
    labels = np.zeros(len(s), dtype=int)
    clusters, cache = ([] for i in range(2))

    for i in range(k):
        v =rand_vectors()
        c = Centroid(v,i)
        CENTROIDS.append(c)
        clusters.append(Cluster([],0.0,c))
        cache.append(0.0)
        print(c.str())                                  #TESTING TODO REMOVE

    error = calc_error(cache)

    #Perform clustering
    while error > THRESHOLD:
        for i in range(len(s)):
            #Map centroid/cluster id to the ith pixel
            cid = label(s[i])
            labels[i] = cid
            clusters[cid].append(s[i])
        cache = [CENTROIDS[i].get_value() for i in range(k)]
        for i in range(k):
            #Update each centroid with the mean cluster value
            CENTROIDS[i].update(clusters[i].get_mean())
            clusters[i] = Cluster([],0.0,CENTROIDS[i])
        error = calc_error(cache)

    return labels


def segmentation(image, features):
    '''
    Manages the operational flow for image segmentation
    '''
    clusters,img = quantize(image, features, K)
    #Reshape back to the original image dimensions
    img = img_reshape(img, features, image, False)
    img = img.astype(np.uint8)
    img = cs_convert(img, features, True)

    name = F+'-k'+str(K)
    write_result(None, img, name)
    print("SSL: "+str(loss(image, img)))


def cli():
    '''
    Command Line Interface that handles optional paramaters for execution.
    Execution: python seg.py -f -k
        -f: feature specification [e.g. RGB,HSV,LAB,GRY]
        -k: number of clusters [e.g. 6]
    @params:
        none
    @returns:
        None
    '''
    if (len(sys.argv) < 2):
        print('Need target image')
        print('Usage: python seg.py <image path> [-f <colorspace>] [-k <clusters>]')
        print('\nOptional flags:')
        print('\t-f colorspace\tRGB,HSV,LAB,GRY')
        print('\t-k clusters\t# of clusters (e.g. 6)')
        print('\nExample:')
        print('\tpython seg.py images/lambo.jpg HSV 7\n')
        sys.exit()
    if (len(sys.argv) > 2):
        global F
        F = sys.argv[2].upper()
    if (len(sys.argv) > 3):
        global K
        K = int(sys.argv[3])

    return sys.argv[1]


def main():
    '''
    Main function handling input, output and the program's operational flow
    @params:
        none
    @returns:
        None
    '''
    path = cli()

    if not os.path.isdir(path) and path.split('.')[1].lower() in EXTENSIONS:
        global PATH
        PATH = path
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        segmentation(img, F)
    else:
        print('Error unsupported input - ' + str(path))
        sys.exit()


main()

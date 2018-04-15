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
################################################################
#Program and Segmentation parameter flags
################################################################
K = 6                      #k-value/clusters
F = 'HSV'                   #Feature usage


class Centroids(object):
    def __init__ (self,centroids):
        self.centroids = centroids
    def get_centroids(self):
        return self.centroids
    def update_all(self,centroids):
        self.centroids = centroids
    def update_one(self,centroid,i):
        self.centroids[i] = centroid


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


def calc_error(centroids, cache):
    '''
    Calculate the average step distance from the prior centroid to the current.
    @params:
        centroids: list of centroid objects
        cache: list of the prior centroid values
    @returns:
        error: the difference between the cluster average and the centroid
    '''
    error = 0.0
    k = len(centroids)
    for i in range(k):
        error += euclidean(centroids[i].value, cache[i], None)
    error = error/k
    print("Cluster error: "+str(error))                 #TESTING TODO REMOVE
    return error


def euclidean(a, b, ax=1):
    '''
    Calculate the euclidean distance between two points
    '''
    return np.linalg.norm(a - b)


def rvectors():
    '''
    Generate random values based on the feature colorspace.
    @params:
        none
    @returns:
        rv: random channel / intensity values
    '''
    rv = np.empty(shape=0)
    for f in FEATURES[F]:
        rv = np.append(rv, np.random.randint(f[0],f[1]))
    return rv


def o_reshape(arr, feature, image, flat):
    '''
    Reshape the array to the designated image dimensions
    '''
    if not flat:
        return arr.reshape((image.shape[0], image.shape[1], len(FEATURES[feature])))
    else:
        return arr.reshape((image.shape[0]*image.shape[1], len(FEATURES[feature])))


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
    #Conditions based on features
    if features == 'GRY':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif features == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif features == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    #Flatten the image & perform kmeans
    samples = o_reshape(image, features, image, True)
    cluster,centroids = kmeans(samples,k)
    cmap = np.zeros(samples.shape)

    #Assign centroid intensity value to each pixel in it's cluster
    for i in range(len(samples)):
        cmap[i] = centroids[cluster[i]].get_value()

    img = cluster.reshape((image.shape[0],image.shape[1],1))#o_reshape(cluster, features, image, False)

    return [img, cmap]


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
    labels = np.zeros(len(s), dtype=int)#[None]*len(s)
    centroids, clusters, cache = ([] for i in range(3))

    for i in range(k):
        v =rvectors()
        c = Centroid(v,i)
        centroids.append(c)
        clusters.append(Cluster([],0.0,c))
        cache.append(0.0)
        print(c.str())                                  #TESTING TODO REMOVE

    error = calc_error(centroids,cache)

    #Perform clustering
    while error > THRESHOLD:
        for i in range(len(s)):
            #Map centroid/cluster id to the ith pixel
            cid = label(s[i],centroids)
            labels[i] = cid#label(s[i],centroids)
            clusters[cid].append(s[i])
        cache = [centroids[i].get_value() for i in range(k)]
        for i in range(k):
            centroids[i].update(clusters[i].get_mean())
            clusters[i] = Cluster([],0.0,centroids[i])
        error = calc_error(centroids,cache)

    return [labels, centroids]


def segmentation(image, features):
    '''
    Manages the operational flow for image segmentation
    '''
    clusters,img = quantize(image, features, K)
    #Reshape back to the original image dimensions
    img = o_reshape(img, features, image, False)
    img = img.astype(int)

    name = F+'-k'+str(K)
    write_result(None, img, name)
    print("SSL: "+str(loss(image, img)))


def cli():
    '''
    Command Line Interface that handles optional paramaters for execution.
    Execution options:
        python seg.py -f -k
        -f: feature specification [e.g. RGB,HSV,LAB,GRY]
        -k: number of clusters [e.g. 6]
    @params:
        none
    @returns:
        None
    '''
    if (len(sys.argv) < 2):
        print('Need target image')
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

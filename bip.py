'''
@Author: Ryan Switzer

Basic image processing script written in python using OpenCV to determine the
dominant color of an image, sort accordingly, plot the original histogram,
perform histogram equalization and then plot the resulting image histogram
(i.e. after equalization). Resulting images of the equalization shall be written
to  disk using the following naming convention <image gname>_eq.<extension>.
'''
import pdb
import cv2
import sys
import os
import math
from matplotlib import pyplot as plt
import numpy as np

#Dictionary to sort images by color dominance
DOMINANT = {'red': [], 'green': [], 'blue': [], 'none': []}
#Set of accepted image formats
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','jpx','j2k','j2c','fpx', \
                  'pcd','pdf','png','ppm','webp','bmp','bpg','dib','wav', \
                  'cgm','svg','gif'])


#Test function to verify color sorting functionality
def test_sort():
    for k, v in DOMINANT.items():
        print(k,v)
        

#Perform histogram equalization, plot it and write resultant image
def histo_equalization(file,img):
    #Convert to HSV to perform histogram equalization
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
    img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
    # Create side-by-side comparison view and write
    result = np.hstack((img,img_eq))
    cv2.imwrite('eqResult_'+file,result)
    
    #Create resulting image name and write
    #newfile = '.'.join(file.split('.')[:-1]) + '_eq.' + file.split('.')[-1]
    #cv2.imwrite(newfile, img_eq)
    
    #Plot equalized image histogram
    create_histo(img_eq)


#Plot
def create_histo(img):
    color = ('b','g','r')
    for i,c in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color = c)
        plt.xlim([0,256])
    plt.show()


#Determine the dominant color and store accordingly
def sort_color(file,img):
    channels = [0,1,2]
    mtp = np.mean(np.mean(img,0),0)
    dom = max(mtp[channels])

    #If more than one channel shares the max value, there is no dominant color
    if np.count_nonzero(mtp == dom) > 1:
        DOMINANT['none'].append(file)
    else:
        for channel in mtp:
            if dom == channel:
                if np.argwhere(mtp == channel) == 0:
                    DOMINANT['blue'].append(file)
                elif np.argwhere(mtp == channel) == 1:
                    DOMINANT['green'].append(file)
                elif np.argwhere(mtp == channel) == 2:
                    DOMINANT['red'].append(file)
    
    
#Main function handling input, output and the programs operational flow
def main():
    if (len(sys.argv) < 2):
        print('Need target directory')
        sys.exit()
    folder = sys.argv[1]
    
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isdir(path) and file.split('.')[1].lower() in EXTENSIONS:
            print(file) #REMOVE TEST
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_UNCHANGED)
            sort_color(file,img)
            
            create_histo(img)
            
            histo_equalization(file,img)
    test_sort()
    
    
main()
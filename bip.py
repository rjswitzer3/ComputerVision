import pdb
import cv2
import sys
import os
import math
from matplotlib import pyplot as plt
import numpy as np

DOMINANT = {'red': [], 'green': [], 'blue': [], 'none': []}
EXTENSIONS = set(['jpg','jpeg','jif','jfif','jp2','jpx','j2k','j2c','fpx', \
                  'pcd','pdf','png','ppm','webp','bmp','bpg','dib','wav', \
                  'cgm','svg','gif'])


def test_sort():
    for k, v in DOMINANT.items():
        print(k,v)

def create_histo(img):
    color = ('b','g','r')
    for i,c in enumerate(color):
        hist = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(hist,color = c)
        plt.xlim([0,256])
    plt.show()


def sort_color(file,img):
    colorspace = cv2.COLOR_BGR2HSV
    channels = [0,1,2]
    mtp = np.mean(np.mean(img,0),0)
    dom = max(mtp[channels])
    
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
    
    create_histo(img)
    
    
#Main function handling input, output and process flow
def main():
    if (len(sys.argv) < 2):
        print('Need target directory')
        sys.exit()
    folder = sys.argv[1]
    
    for file in os.listdir(folder):
        if file.split('.')[1].lower() in EXTENSIONS:
            img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_UNCHANGED)
            sort_color(file,img)
    test_sort()
    
    
main()
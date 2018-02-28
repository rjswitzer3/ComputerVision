'''
@Author Ryan Switzer

Personal implemnetation of the canny edge detection algorithm using OpenCV.
'''
import cv2
import numpy as np
import sys
import os
import math
from scipy.signal import convolve2d


def ritconv():
    # TODO: Implement me


def ritgrad():
    # TODO Implement me


def ritcan():
    # TODO Implement me


def main():
    if (len(sys.argv) < 2):
        print('Need target file') # TODO Do we need to process multiple imgs?
        sys.exit()
    img = sys.argv[1]

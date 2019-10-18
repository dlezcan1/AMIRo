import cv2
import numpy as np
from os.path import exists

ROI = [150,980,400,565]          #xleft, xright, ytop, ybottom #Region of Interest
bor1 = [0,26,93,165]             #xleft, xright, ytop, ybottom #for the bottomleft corner, blackout
bor2 = [0,29,0,58]               #xleft, xright, ytop, ybottom #for the topleft corner, blackout

#used to determine bounding box for ROI, one-time use
def mouse_callback_pointposition(event,c,y,flags,param):
    print('x:{}px y:{}px'.format(x,y))

def gen_kernel(shape):
    """Function to generate the shape of the kernel for image processing

    @param shape: 2-tuple (a,b) of integers for the shape of the kernel
    @return: returns axb numpy array of value of 1's of type uint8
"""
    return np.ones(shape,np.uint8)

#gen_kernel

def segment_needle(filename,seg_method):
    """Function that will segment the needle given from an image filename through
    thresholding or canny edge detection.
    @param filename: image filename
    @param seg_method: string that can be either "thresh" or "canny"

    @return: segmented image

    @raise NotImplementedError: raises exception if seg_method not "thresh" or "canny"
    @raise FileNotFoundError: raises expection if the filename was not found
    @raise TypeError: raises exception if the file was unreadable by openCV
    """
    seg_method = seg_method.lower() #lower case the segmentation method
    
    if not(exists(filename)):
        raise FileNotFoundError("File was not found: {}".format(filename))
    
    full_img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

    if type(full_img) == type(None):
        raise TypeError("{} is not an image file readable by openCV.")
    
    img = full_img[ROI[2]:ROI[3],ROI[0]:ROI[1]]
    img = cv2.bilateralFilter(img,9,65,65)
    cv2.imshow(filename,img)

    if seg_method == "thresh": #thresholding
        _, thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)

##        cv2.imshow('Threshold before artifact removal',thresh)
        ## Remove (pre-determined for simplicity in this code) artifacts manually
        ## I plan to make this part of the algorithm to be incorproated into GUI
        thresh[bor1[2]:bor1[3],bor1[0]:bor1[1]] = 0
        thresh[bor2[2]:bor2[3],bor2[0]:] = 0
##        cv2.imshow('Threshold after artifact removal',thresh)

        kernel = gen_kernel((9,9))
        thresh_fixed = cv2.dilate(thresh,kernel,iterations=2)
        kernel = gen_kernel((11,31))
        thresh_fixed = cv2.erode(thresh_fixed,kernel,iterations=1)
        kernel = gen_kernel((7,7))
        thresh_fixed = cv2.morphologyEx(thresh_fixed,cv2.MORPH_OPEN,kernel)
        thresh_fixed = cv2.erode(thresh_fixed,kernel,iterations=1)

##        cv2.imshow('Threshold_fixed',thresh_fixed)

        retval = thresh_fixed
        retval = thresh

    #if

    elif seg_method == "canny": #canny
        ## Canny Filtering for Edge detection
        canny1 = cv2.Canny(img,25,255)
##        cv2.imshow('canny1 before',canny1)
        ## Remove (pre-determined for simplicity in this code) artifacts manually
        ## I plan to make this part of the algorithm to be incorproated into GUI
        canny1[bor1[2]:bor1[3],bor1[0]:bor1[1]] = 0
        canny1[bor2[2]:bor2[3],bor2[0]:] = 0
##        cv2.imshow('canny1 after',canny1)

        # worked for black background
        kernel = gen_kernel((7,7))
        canny1_fixed = cv2.morphologyEx(canny1,cv2.MORPH_CLOSE,kernel)
        kernel = gen_kernel((9,9))
        canny1_fixed = cv2.dilate(canny1_fixed,kernel,iterations=2)
        kernel = gen_kernel((11,31))
        canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)
        kernel = gen_kernel((7,7))
        canny1_fixed = cv2.morphologyEx(canny1_fixed,cv2.MORPH_OPEN,kernel)
        canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)

        retval = canny1_fixed

    #elif

    else:
        raise NotImplementedError('Segmentation method, "{}", has not been implemented.'.format(seg_method))

##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    return retval

#segment_needle







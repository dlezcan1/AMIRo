import cv2
import numpy as np
from skimage.morphology import skeletonize
from os.path import exists
import matplotlib.pyplot as plt

ROI = [40,975,420,540]          #xleft, xright, ytop, ybottom #Region of Interest
top_bor = 35                    # x  0:top_bor      -> 0
br_bor = [905, 82]              # [(1):, (0):]      -> 0
tr_bor = [910, -1, 0, 72]       # [(2):(3): (0):(1) -> 0
tip_bor = 915                   # y tip_bor:end     -> 0

#used to determine bounding box for ROI, one-time use
def mouse_callback_pointposition(event,x,y,flags,param):
    print('x:{}px y:{}px'.format(x,y))

def gen_kernel(shape):
    """Function to generate the shape of the kernel for image processing

    @param shape: 2-tuple (a,b) of integers for the shape of the kernel
    @return: returns axb numpy array of value of 1's of type uint8
"""
    return np.ones(shape,np.uint8)

#gen_kernel

def generate_functions(bool_print: bool):
    """Determines if we want the image processig functions to display step-by-step or not"""
    if bool_print:
        pass
    else:
        pass

# generate_functions

def find_coordinate_image(img):
    cv2.imshow('Mouse Coordinate Find',img)
    cv2.setMouseCallback('Mouse Coordinate Find',
                         mouse_callback_pointposition)  #used to determine bounding box for ROI
    cv2.waitKey(0)
    cv2.destroyAllWindows

# find_coordinate_image

def segment_needle(filename,seg_method, bool_show: bool = False):
    """Function that will segment the needle given from an image filename through
    thresholding or canny edge detection.
    @param filename:   image filename
    @param seg_method: string that can be either "thresh" or "canny"
    @param bool_show:  a boolean on whether you want to show img processing steps

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
    ROI_image = img
    img = cv2.bilateralFilter(img,9,65,65)
##    cv2.imshow(filename,img)

    if seg_method == "thresh": #thresholding
        _, thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)

##        cv2.imshow('Threshold before artifact removal',thresh)
        ## Remove (pre-determined for simplicity in this code) artifacts manually
        ## I plan to make this part of the algorithm to be incorproated into GUI
        thresh[0:top_bor, :] = 0
        thresh[br_bor[1]:,br_bor[0]:] = 0
        thresh[tr_bor[2]:tr_bor[3], tr_bor[0]:] = 0
        thresh[:,tip_bor:] = 0
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
##        cv2.imshow("just canny",canny1)
        if bool_show:
            cv2.imshow('canny1 before',canny1)
        ## Remove (pre-determined for simplicity in this code) artifacts manually
        ## I plan to make this part of the algorithm to be incorproated into GUI
        canny1[0:top_bor, :] = 0
        canny1[br_bor[1]:,br_bor[0]:] = 0
        canny1[tr_bor[2]:tr_bor[3], tr_bor[0]:] = 0
        canny1[:,tip_bor:] = 0
        if bool_show:
            cv2.imshow('canny1 after',canny1)

        # worked for black background
        kernel = gen_kernel((7,7))
        canny1_fixed = cv2.morphologyEx(canny1,cv2.MORPH_CLOSE,kernel)
        if bool_show:
            cv2.imshow('7x7 Closed',canny1_fixed)
        kernel = gen_kernel((9,9))
        canny1_fixed = cv2.dilate(canny1_fixed,kernel,iterations=2)
        if bool_show:
            cv2.imshow('9x9 dilated x2',canny1_fixed)
        kernel = gen_kernel((11,31))
        canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)
        if bool_show:
            cv2.imshow('11x31 eroded x1', canny1_fixed)
        kernel = gen_kernel((7,7))
        canny1_fixed = cv2.morphologyEx(canny1_fixed,cv2.MORPH_OPEN,kernel)
        if bool_show:
            cv2.imshow('7x7 Opened', canny1_fixed)
        canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)
        if bool_show:
            cv2.imshow('7x7 eroded x2 (finished)', canny1_fixed)

        retval = canny1_fixed

        retval[retval > 0] = 1
        retval = skeletonize(retval)
        
        retval = retval.astype(np.uint8)
        retval[retval > 0] = 255
        if bool_show:
            cv2.imshow('Skeletonized', retval)

        

    #elif

    else:
        raise NotImplementedError('Segmentation method, "{}", has not been implemented.'.format(seg_method))

##    cv2.waitKey(0)
##    cv2.destroyAllWindows()

    return ROI_image, retval

#segment_needle

def fit_polynomial_skeleton(img_skeleton: np.ndarray, order: int):

    assert(np.max(img_skeleton) <= 1)

    N_rows, N_cols = np.shape(img_skeleton)

    x = np.arange(N_cols)         # x-coords
    y = N_rows*np.ones(N_cols)    # y-coords
    y = np.argmax(img_skeleton,0)

    x = x[y > 0]
    y = y[y > 0]

    if len(x) == 0:
        x = np.zeros(N_cols)
        y = np.zeros(N_cols)
        p = np.poly1d([0])
    else:
        p = np.poly1d(np.polyfit(x,y,order))

    return x, y, p

# fit_polynomial_skeleton


def get_curvature(filename: str, poly_deg: int, active_areas):
    seg_needle = segment_needle(filename,'canny')
    thresh_needle = np.copy(seg_needle)
    thresh_neede[thresh_needle > 0] = 1

    x, y, p = fit_polynomial_skeleton(thresh_needle,10)

    p_2deriv = np.polyder(p,2)
    p_1deriv = np.polyder(p,1)

    curvature = (p_2deriv(active_areas))/((1 + p_1deriv)(active_areas))

    return curvature

# get_curvature




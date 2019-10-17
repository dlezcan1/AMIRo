import cv2
import numpy as np

ROI = [150,980,400,565]          #xleft, xright, ytop, ybottom #Region of Interest
bor1 = [0,26,93,165]             #xleft, xright, ytop, ybottom #for the bottomleft corner, blackout
bor2 = [0,29,0,58]               #xleft, xright, ytop, ybottom #for the topleft corner, blackout

#used to determine bounding box for ROI, one-time use
def mouse_callback_pointposition(event,x,y,flags,param):
    print('x:{}px y:{}px'.format(x,y))

#start script
directory = 'C:\\Users\\dlezcan1\\Documents\\DL-Sharing-Folder\\'

## Read in images
file1 = 'image1-test-color.png'                 #without cardboard cover
file2 = 'image2-test-color-cardboardcover.png'  #with the cardboard cover
file3 = 'image3-test-color-s_shape.png'         #s-shape bend
file4 = 'image4-test-color-c_shape.png'         #now has black background

full_img = cv2.imread(directory+file1,cv2.IMREAD_GRAYSCALE)
##full_img = cv2.imread(directory+file2,cv2.IMREAD_GRAYSCALE)
full_img = cv2.imread(directory+file3,cv2.IMREAD_GRAYSCALE)
full_img = cv2.imread(directory+file4,cv2.IMREAD_GRAYSCALE)


img = full_img[ROI[2]:ROI[3],ROI[0]:ROI[1]]


cv2.imshow('Full Image',full_img)
cv2.imshow('Image ROI',img)

##cv2.setMouseCallback('image1',mouse_callback_pointposition)#used to determine bounding box for ROI
##cv2.waitKey(0)

#smooth the image for noise without removing edges
img = cv2.bilateralFilter(img,9,65,65)


## Canny Filtering for Edge detection
canny1 = cv2.Canny(img,25,255)
cv2.imshow('canny1 before',canny1)
## Remove (pre-determined for simplicity in this code) artifacts manually
## I plan to make this part of the algorithm to be incorproated into GUI
canny1[bor1[2]:bor1[3],bor1[0]:bor1[1]] = 0
canny1[bor2[2]:bor2[3],bor2[0]:] = 0
cv2.imshow('canny1 after',canny1)

"""
## first attempt @ segmentation
kernel = np.ones((9,9),np.uint8)
canny1_dilated = cv2.dilate(canny1,kernel,iterations=2)
canny1_fixed = cv2.erode(canny1_dilated,kernel,iterations=3)
canny1_fixed = cv2.morphologyEx(canny1_fixed,cv2.MORPH_CLOSE,kernel)
kernel = np.ones((3,3),np.uint8)
canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)
"""
## More Robust Segmentation Procedure
#connect broken pieces
kernel = np.ones((15,15),np.uint8)
canny1_fixed = cv2.dilate(canny1,kernel,iterations=2)
canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=2)

#thin out
kernel = np.ones((3,3),np.uint8)
canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=2)
canny1_median = cv2.medianBlur(canny1_fixed,5)

#results of processing
cv2.imshow('canny1',canny1)
cv2.imshow('canny1_fixed',canny1_fixed)
cv2.imshow('canny1_median',canny1_median)

cv2.waitKey(0)
cv2.destroyAllWindows()





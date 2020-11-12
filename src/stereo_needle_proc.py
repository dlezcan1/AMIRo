'''
Created on Nov 6, 2020

This is a file for building image processing to segment the needle in stereo images


@author: dlezcan1

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors as pltcolors
from skimage.morphology import skeletonize, thin, medial_axis
from sklearn.cluster import MeanShift, estimate_bandwidth

# color HSV ranges
COLOR_HSVRANGE_RED = ( ( 0, 50, 50 ), ( 10, 255, 255 ) )
COLOR_HSVRANGE_BLUE = ( ( 110, 50, 50 ), ( 130, 255, 255 ) )
COLOR_HSVRANGE_GREEN = ( ( 40, 50, 50 ), ( 75, 255, 255 ) )
COLOR_HSVRANGE_YELLOW = ( ( 25, 50, 50 ), ( 35, 255, 255 ) )


def blackout( img, tl, br ):
    img[tl[0]:br[0], tl[1]:br[1]] = 0
    
    return img
    
# blackout


def blackout_regions( img, bo_regions: list ):
    
    for tl, br in bo_regions:
        img = blackout( img, tl, br )
        
    # for
    
    return img

# blackout_regions


def bin_close( left_bin, right_bin, ksize = ( 6, 6 ) ):
    kernel = np.ones( ksize )
    left_close = cv2.morphologyEx( left_bin, cv2.MORPH_CLOSE, kernel )
    right_close = cv2.morphologyEx( right_bin, cv2.MORPH_CLOSE, kernel )
    
    return left_close, right_close

# bin_close


def bin_dilate( left_bin, right_bin, ksize = ( 3, 3 ) ):
    kernel = np.ones( ksize )
    left_dil = cv2.dilate( left_bin, kernel )
    right_dil = cv2.dilate( right_bin, kernel )
    
    return left_dil, right_dil

# bin_dilate


def bin_open( left_bin, right_bin, ksize = ( 6, 6 ) ):
    kernel = np.ones( ksize )
    left_open = cv2.morphologyEx( left_bin, cv2.MORPH_OPEN, kernel )
    right_open = cv2.morphologyEx( right_bin, cv2.MORPH_OPEN, kernel )
    
    return left_open, right_open

# bin_open


def canny( left_img, right_img, lo_thresh = 150, hi_thresh = 200 ):
    ''' Canny edge detection '''

    canny_left = cv2.Canny( left_img, lo_thresh, hi_thresh )
    canny_right = cv2.Canny( right_img, lo_thresh, hi_thresh )
    
    return canny_left, canny_right

# canny


def color_segmentation( left_img, right_img, color ):
    ''' get the pixels of a specific color'''
    # testing on red only
    if color.lower() == 'red':
        lb = COLOR_HSVRANGE_RED[0] 
        ub = COLOR_HSVRANGE_RED[1] 
    
    # if
    
    # convert into HSV color space 
    left_hsv = cv2.cvtColor( left_img, cv2.COLOR_BGR2HSV )
    right_hsv = cv2.cvtColor( right_img, cv2.COLOR_BGR2HSV )
    
    # determine which colors are within the bounds
    left_mask = cv2.inRange( left_hsv, lb, ub )
    right_mask = cv2.inRange( right_hsv, lb, ub )
    
    # masked images
    left_color = cv2.bitwise_and( left_img, left_img, mask = left_mask )
    right_color = cv2.bitwise_and( right_img, right_img, mask = right_mask )
    
    return left_mask, right_mask, left_color, right_color
    
# color_segmentation
    

def contours( left_skel, right_skel ):
    conts_l, *_ = cv2.findContours( left_skel.astype( np.uint8 ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
    conts_r, *_ = cv2.findContours( right_skel.astype( np.uint8 ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
    
    conts_l = sorted( conts_l, key = len, reverse = True )
    conts_r = sorted( conts_r, key = len, reverse = True )
    
    return conts_l, conts_r

# contours


def gridproc_stereo( left_img, right_img,
                     bor_l: list = [], bor_r: list = [],
                     proc_show: bool = False ):
    ''' wrapper function to segment the grid out of a stereo pair '''
    # TODO
    
    # convert to grayscale if not already
    if left_img.ndim > 2:
        left_img = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
        right_img = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )

    # if
    
    # start the image qprocessing
    left_thresh, right_thresh = thresh( 2 * left_img, 2 * right_img )
    
    left_thresh_bo = blackout_regions( left_thresh, bor_l )
    right_thresh_bo = blackout_regions( right_thresh, bor_r )
    
    left_med, right_med = median_blur( left_thresh_bo, right_thresh_bo, 5 )
    
    left_canny, right_canny = canny( left_med, right_med, 180, 200 )
    
    # harris corner detection
    left_centroid = cv2.cornerHarris( left_thresh_bo, 2, 5, 0.04 )
    left_centroid = cv2.dilate( left_centroid, None )
    _, left_corners = cv2.threshold( left_centroid, 0.01 * left_centroid.max(),
                                255, 0 )
    left_corners = np.int0( left_corners )
     
    right_centroid = cv2.cornerHarris( right_thresh_bo, 2, 5, 0.04 )
    right_centroid = cv2.dilate( right_centroid, None )
    _, right_corners = cv2.threshold( right_centroid, 0.01 * right_centroid.max(),
                                255, 0 )
    right_corners = np.int0( right_corners )
    
    left_crnr = cv2.cvtColor( left_img, cv2.COLOR_GRAY2RGB )
    right_crnr = cv2.cvtColor( right_img, cv2.COLOR_GRAY2RGB )
    left_crnr[left_corners] = [255, 0, 0]
    right_crnr[right_corners] = [255, 0, 0]
    
    # plotting
    if proc_show:
        plt.ion()
        
        plt.figure()
        plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap = 'gray' )
        plt.title( 'adaptive thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap = 'gray' )
        plt.title( 'region suppression: after thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_med, right_med, 150 ), cmap = 'gray' )
        plt.title( 'median: after region suppression' )
        
        plt.figure()
        plt.imshow( imconcat( left_canny, right_canny, 150 ), cmap = 'gray' )
        plt.title( 'canny: after median filtering' )
        
        plt.figure()
        plt.imshow( imconcat( left_centroid, right_centroid ), cmap = 'gray' )
        plt.title('dilated harris corner respon0se')
        
        plt.figure()
        plt.imshow( imconcat( left_crnr, right_crnr ) )
        plt.title( 'corner detection' )
        
    # if

# gridproc_stereo


def houghlines( left_img, right_img ):
    ''' function for performing hough lines transform on a stereo pair '''
    
    # TODO
    
# houghlines
    

def hough_quadratic( img ):
    ''' hough transform to fit a quadratic 
    
        Want to fit a function y = a x**2 + b x + c
        
        This will pickout the SINGLE argmax
    
    '''
    raise NotImplementedError( 'hough_quadratic is not yet implemented.' )

# hough_quadratic
    

def imconcat( left_im, right_im, pad_val = 0 ):
    ''' wrapper for concatenating images'''
    
    if left_im.ndim == 2:
        pad_left_im = np.pad( left_im, ( ( 0, 0 ), ( 0, 20 ) ), constant_values = pad_val )
    
    elif left_im.ndim == 3:
        pad_left_im = np.pad( left_im, ( ( 0, 0 ), ( 0, 0 ), ( 0, 0 ) ), constant_values = pad_val )
    
    return np.concatenate( ( pad_left_im, right_im ), axis = 1 )

# imconcat


def median_blur( left_thresh, right_thresh, ksize = 11 ):
    left_med = cv2.medianBlur( left_thresh, ksize )
    right_med = cv2.medianBlur( right_thresh, ksize )
    
    return left_med, right_med

# median_blur


def meanshift( left_bin, right_bin, q = 0.3, n_samps:int = 200, plot_lbls:bool = False ):
    # get non-zero coordinates
    yl, xl = np.nonzero( left_bin )
    pts_l = np.vstack( ( xl, yl ) ).T
    
    yr, xr = np.nonzero( right_bin )
    pts_r = np.vstack( ( xr, yr ) ).T
    
    # estimate meanshift bandwidth
    bandwidth_l = estimate_bandwidth( pts_l, quantile = q, n_samples = n_samps )
    bandwidth_r = estimate_bandwidth( pts_r, quantile = q, n_samples = n_samps )
    
    # meanshift fit
    ms_l = MeanShift( bandwidth = bandwidth_l, bin_seeding = True )
    ms_r = MeanShift( bandwidth = bandwidth_r, bin_seeding = True )
    
    ms_l.fit( pts_l )
    ms_r.fit( pts_r )
    
    # get the labels
    left_lbls = ms_l.labels_
    right_lbls = ms_r.labels_  
    
    # plot the clusters
    cols = list( pltcolors.CSS4_COLORS.values() )
    if plot_lbls:
        # plot left
        nlbls = len( np.unique( left_lbls ) )

        plt.figure()
        for i, c in zip( range( nlbls ), cols[:nlbls] ):
            members = ( left_lbls == i )
            
            plt.plot( xl[members], yl[members], 'o', color = c, markersize = 12 )
            
        # for
        plt.gca().invert_yaxis()

        # plot right
        nlbls = len( np.unique( right_lbls ) )

        plt.figure()
        for i, c in zip( range( nlbls ), cols[:nlbls] ):
            members = ( right_lbls == i )
            
            plt.plot( xr[members], yr[members], 'o', color = c, markersize = 12 )
            
        # for
        plt.gca().invert_yaxis()

        plt.show()
        
    # if
    
    return ( pts_l, left_lbls ), ( pts_r, right_lbls )

# meanshift


def needleproc_stereo( left_img, right_img,
                         bor_l:list = [], bor_r:list = [],
                         proc_show: bool = False ):
    ''' wrapper function to process the left and right image pair for needle
        centerline identification
        
     '''
    
    # convert to grayscale if not already
    if left_img.ndim > 2:
        left_img = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
        right_img = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )

    # if
    
    # start the image qprocessing
    left_thresh, right_thresh = thresh( left_img, right_img )
    
    left_thresh_bo = blackout_regions( left_thresh, bor_l )
    right_thresh_bo = blackout_regions( right_thresh, bor_r )
    
    left_tmed, right_tmed = median_blur( left_thresh_bo, right_thresh_bo, ksize = 5 )
    
    left_open, right_open = bin_open( left_tmed, right_tmed, ksize = ( 7, 7 ) )
    
    left_close, right_close = bin_close( left_open, right_open, ksize = ( 16, 16 ) )
    
    left_dil, right_dil = bin_dilate( left_close, right_close, ksize = ( 0, 0 ) )
    
    left_skel, right_skel = skeleton( left_dil, right_dil )
    
    # get the contours ( sorted by length)
    conts_l, conts_r = contours( left_skel, right_skel )
    
    if proc_show:
        plt.ion()
        
        plt.figure()
        plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap = 'gray' )
        plt.title( 'adaptive thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap = 'gray' )
        plt.title( 'region suppression: after thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_tmed, right_tmed, 150 ), cmap = 'gray' )
        plt.title( 'median filtering: after region suppression' )
        
        plt.figure()
        plt.imshow( imconcat( left_close, right_close, 150 ), cmap = 'gray' )
        plt.title( 'opening: after median' )
        
        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap = 'gray' )
        plt.title( 'closing: after opening' )

        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap = 'gray' )
        plt.title( 'dilation: after closing' )
        
        plt.figure()
        plt.imshow( imconcat( left_skel, right_skel, 150 ), cmap = 'gray' )
        plt.title( 'skeletization: after dilation' )
        
        cont_left = left_img.copy().astype( np.uint8 )
        cont_right = right_img.copy().astype( np.uint8 )
        
        cont_left = cv2.cvtColor( cont_left, cv2.COLOR_GRAY2RGB )
        cont_right = cv2.cvtColor( cont_right, cv2.COLOR_GRAY2RGB )
        
        cv2.drawContours( cont_left, conts_l, 0, ( 255, 0, 0 ), 3 )
        cv2.drawContours( cont_right, conts_r, 0, ( 255, 0, 0 ), 3 )
        
        plt.figure()
        plt.imshow( imconcat( cont_left, cont_right, 150 ), cmap = 'gray' )
        plt.title( 'contour: the longest 1' )
        
        plt.show()
        while True:
            if plt.waitforbuttonpress( 0 ):
                break
            
        # while
        plt.close( 'all' )
        
    # if
    
    return left_skel, right_skel, conts_l, conts_r
    
# needleproc_stereo


def roi( img, roi, full:bool = False ):
    ''' return region of interest 
    
        @param roi: [tuple of top-left point, tuple of bottom-right point]
        
        @return: subimage of the within the roi
    '''
    # TODO
    
    tl_i, tl_j = roi[0]
    br_i, br_j = roi[1]
    
    # zero-out value
    zval = 0 if img.ndim == 2 else np.array( [0, 0, 0] )
    
    if full:
        img_roi = img.copy()
        
        # zero out values
        img_roi [:tl_i, :] = zval
        img_roi [br_i + 1:, :] = zval
        
        img_roi [:, :tl_j] = zval
        img_roi [:, br_j + 1:] = zval
        
    # if
        
    else:
        img_roi = img[tl_i:br_i, tl_j:br_j].copy()
        
    # else
        
    return img_roi 

# roi


def skeleton( left_bin, right_bin ):
    ''' skeletonize the left and right binary images'''
    
    left_bin = ( left_bin > 0 ).astype( np.uint8 )
    right_bin = ( right_bin > 0 ).astype( np.uint8 )
    
    left_skel = skeletonize( left_bin )
    right_skel = skeletonize( right_bin )
#     left_skel = thin( left_bin, max_iter = 2 )
#     right_skel = thin( right_bin, max_iter = 2 )
    
    return left_skel, right_skel

# skeleton


def stereomatch_needle( left_conts, right_conts, start_location = "tip", axis = 1 ):
    ''' stereo matching needle arclength points for the needle
        
        
        Args:
            (left/right)_conts: a nx2 array of pixel coordinates
                                for the contours in the (left/right) image
            
            start_location (Default: "tip"): a string of where to start counting.
                                             tip is only implemented.
    
     '''
    n = min( left_conts.shape[0], right_conts.shape[0] )
    
    if start_location.lower() == "tip":
        left_idx = np.argsort( left_conts[:, axis] )[-n:]
        right_idx = np.argsort( right_conts[:, axis] )[-n:]
        
        left_matches = left_conts[left_idx]
        right_matches = right_conts[right_idx]
        
    # if
    
    else:
        raise ValueError( f"start_location = {start_location} not valid." )
    
    return left_matches, right_matches

# stereomatch_needle


def thresh( left_img, right_img ):
    ''' image thresholding'''
    left_thresh = cv2.adaptiveThreshold( left_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 4 )
    right_thresh = cv2.adaptiveThreshold( right_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 4 )
    
    return left_thresh, right_thresh

# thresh


def main_dbg():
    # directory settings
    stereo_dir = "../Test Images/stereo_needle/"
    needle_dir = stereo_dir + "needle_examples/"
    grid_dir = stereo_dir + "grid_only/"
    
    # the left and right image to test
    num = 5
    left_fimg = needle_dir + f"left-{num:04d}.png"
    right_fimg = needle_dir + f"right-{num:04d}.png"
    
    left_img = cv2.imread( left_fimg, cv2.IMREAD_COLOR )
    right_img = cv2.imread( right_fimg, cv2.IMREAD_COLOR )
    left_gray = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
    right_gray = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )
    
    # blackout regions
    bor_l = [( left_gray.shape[0] - 100, 0 ), left_gray.shape]
    bor_r = bor_l
    
    # perform image processing tests
    left_thresh, right_thresh = thresh( left_gray, right_gray )
    left_tmed, right_tmed = median_blur( left_thresh, right_thresh, ksize = 5 )
    left_open, right_open = bin_open( left_tmed, right_tmed, ksize = ( 7, 7 ) )
    left_close, right_close = bin_close( left_open, right_open, ksize = ( 16, 16 ) )
    left_dil, right_dil = bin_dilate( left_close, right_close, ksize = ( 0, 0 ) )
    left_skel, right_skel = skeleton( left_dil, right_dil )
    left_skel = blackout( left_skel, bor_l[0], bor_l[1] )
    right_skel = blackout( right_skel, bor_r[0], bor_r[1] )
    
    conts_l, conts_r = contours( left_skel, right_skel )
    
    # contour drawing
    print( 'contours length: ', len( conts_l ), len( conts_r ) )
    print( 'left contours:', type( conts_l ) )
    left_conts = np.zeros( left_skel.shape )
    right_conts = np.zeros( right_skel.shape )
    cv2.drawContours( left_conts, conts_l[0:2], -1, ( 255, 255, 255 ), 3 )
    cv2.drawContours( right_conts, conts_r[0:2], -1, ( 255, 255, 255 ), 3 )
    
    # plot finished contour
    left_im = left_img.copy()
    right_im = right_img.copy()
    cv2.drawContours( left_im, conts_l[0:3], 0, ( 255, 0, 0 ), 3 )
    cv2.drawContours( right_im, conts_r[0:3], 0, ( 255, 0, 0 ), 3 )
    
    plt.ion()
    plt.figure()
    plt.imshow( imconcat( left_gray, right_gray ), cmap = 'gray' )
    plt.title( 'images' )

#     plt.figure()
#     plt.imshow( imconcat( left_canny, right_canny, 255 ), cmap = 'gray' )
#     plt.title( 'canny: post median' )
    
#     plt.figure()
#     plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap = 'gray' )
#     plt.title( 'threshold' )
     
    plt.figure()
    plt.imshow( imconcat( left_tmed, right_tmed, 150 ), cmap = 'gray' )
    plt.title( 'median: post threshold' )
     
    plt.figure()
    plt.imshow( imconcat( left_open, right_open, 150 ), cmap = 'gray' )
    plt.title( 'opening: post median' )
    
    plt.figure()
    plt.imshow( imconcat( left_close, right_close, 150 ), cmap = 'gray' )
    plt.title( 'closing: post opening' )
    
    plt.figure()
    plt.imshow( imconcat( left_dil, right_dil, 150 ), cmap = 'gray' )
    plt.title( 'dilation: post closing' )
    
    plt.figure()
    plt.imshow( imconcat( left_skel, right_skel, 150 ), cmap = 'gray' )
    plt.title( 'skeletonize: post closing' )
    
    plt.figure()
    plt.imshow( imconcat( left_conts, right_conts, 125 ), cmap = 'gray' )
    plt.title( 'contours: post skeletonization' )
    
    plt.figure()
    plt.imshow( imconcat( left_im, right_im ) )
    plt.title( 'annotated stereo img proc' )
    
    # close on enter
    plt.show()
    while True:
        if plt.waitforbuttonpress( 0 ):
            break
        
    # while
    
    plt.close( 'all' )
    
# main_dbg


def main_needleproc( file_num, img_dir, save_dir ):
    ''' main method for segmenting the needle centerline in stereo images'''
    # the left and right image to test
    num = 5
    left_fimg = img_dir + f"left-{num:04d}.png"
    right_fimg = img_dir + f"right-{num:04d}.png"
    
    left_img = cv2.imread( left_fimg, cv2.IMREAD_COLOR )
    right_img = cv2.imread( right_fimg, cv2.IMREAD_COLOR )
    left_gray = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
    right_gray = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )
    
    # blackout regions
    bor_l = [( left_gray.shape[0] - 100, 0 ), left_gray.shape]
    bor_r = bor_l
    
    left_skel, right_skel, conts_l, conts_r = needleproc_stereo( left_img, right_img,
                                                                 [bor_l], [bor_r],
                                                                 proc_show = False )
    
    left_cont = left_img.copy()
    left_cont = cv2.drawContours( left_cont, conts_l, 0, ( 255, 0, 0 ), 3 )
    
    right_cont = right_img.copy()
    right_cont = cv2.drawContours( right_cont, conts_r, 0, ( 255, 0, 0 ), 3 )
    
    # save the processed images
    save_fbase = save_dir + f"left-right-{num:04d}" + "_{:s}.png"
    # # skeletons
    plt.imsave( save_fbase.format( 'skel' ), imconcat( left_skel, right_skel, 150 ),
                cmap = 'gray' )
    print( 'Saved figure:', save_fbase.format( 'skel' ) )
    
    # # contours
    plt.imsave( save_fbase.format( 'cont' ), imconcat( left_cont, right_cont ) )
    print( 'Saved figure:', save_fbase.format( 'cont' ) )

# main_needleproc


def main_gridproc( num, img_dir, save_dir ):
    ''' main method to segment the grid in a stereo pair of images'''
    # the left and right image to test    
    left_fimg = img_dir + f"left-{num:04d}.png"
    right_fimg = img_dir + f"right-{num:04d}.png"
    
    left_img = cv2.imread( left_fimg, cv2.IMREAD_COLOR )
    right_img = cv2.imread( right_fimg, cv2.IMREAD_COLOR )
    left_gray = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
    right_gray = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )
    left_img2 = cv2.cvtColor( left_img, cv2.COLOR_RGB2BGR )
    right_img2 = cv2.cvtColor( right_img, cv2.COLOR_RGB2BGR )
    
    # TODO
    # color segmentation
    lmask, rmask, lcolor2, rcolor2 = color_segmentation( left_img2, right_img2, "red" )
    lcolor = cv2.cvtColor( lcolor2, cv2.COLOR_BGR2RGB )
    rcolor = cv2.cvtColor( rcolor2, cv2.COLOR_BGR2RGB )
    
    # find the grid
    gridproc_stereo( left_gray, right_gray, proc_show = True )
    
    # plotting
    plt.ion()
    plt.figure()
    plt.imshow( imconcat( left_img, right_img ) )
    plt.title( 'Original images' )
    
#     plt.figure()
#     plt.imshow( imconcat( lmask.astype( np.uint8 ), rmask.astype( np.uint8 ), 150 ), cmap = 'gray' )
#     plt.title( 'red mask' )
    
    plt.figure()
    plt.imshow( imconcat( lcolor, rcolor ) )
    plt.title( 'masked red color' )
    
    # close on enter
    plt.show()
    while True:
        if plt.waitforbuttonpress( 0 ):
            break
        
    # while
    
    plt.close( 'all' )

# main_gridproc


if __name__ == '__main__':
    # directory settings
    stereo_dir = "../Test Images/stereo_needle/"
    needle_dir = stereo_dir + "needle_examples/"
    grid_dir = stereo_dir + "grid_only/"
    
#     main_needleproc( 5, needle_dir, needle_dir )
    
    main_gridproc( 2, needle_dir, needle_dir )
    
    print( 'Program complete.' )

# if


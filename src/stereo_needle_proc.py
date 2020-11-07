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


def roi( img, roi ):
    ''' return region of interest 
    
        @param roi: [tuple of top-left point, tuple of bottom-right point]
    '''
    # TODO
    pass

# roi


def blackout( img, tl, br ):
    img[tl[0]:br[0], tl[1]:br[1]] = 0
    
    return img
    
# blackout


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


def contours( left_skel, right_skel ):
    conts_l, *_ = cv2.findContours( left_skel.astype( np.uint8 ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
    conts_r, *_ = cv2.findContours( right_skel.astype( np.uint8 ), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE )
    
    conts_l = sorted( conts_l, key = len, reverse = True )
    conts_r = sorted( conts_r, key = len, reverse = True )
    
    return conts_l, conts_r

# contours


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


def thresh( left_img, right_img ):
    ''' image thresholding'''
    left_thresh = cv2.adaptiveThreshold( left_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 4 )
    right_thresh = cv2.adaptiveThreshold( right_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 13, 4 )
    
    return left_thresh, right_thresh

# thresh


if __name__ == '__main__':
    # directory settings
    stereo_dir = "../Test Images/stereo_needle/"
    needle_dir = stereo_dir + "needle_examples/"
    grid_dir = stereo_dir + "grid_only/"
    
    # the left and right image to test
    num = 5
    left_fimg = needle_dir + f"left-{num:04d}.png"
    right_fimg = needle_dir + f"right-{num:04d}.png"
    
    left_gray = cv2.imread( left_fimg, cv2.IMREAD_GRAYSCALE )
    right_gray = cv2.imread( right_fimg, cv2.IMREAD_GRAYSCALE )
    
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
    left_im = cv2.cvtColor( left_gray, cv2.COLOR_GRAY2BGR )
    right_im = cv2.cvtColor( right_gray, cv2.COLOR_GRAY2BGR )
    cv2.drawContours( left_im, conts_l[0:3], -1, ( 0, 0, 255 ), 3 )
    cv2.drawContours( right_im, conts_r[0:3], -1, ( 0, 0, 255 ), 3 )
    
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
    
    print( 'Program complete.' )

# if


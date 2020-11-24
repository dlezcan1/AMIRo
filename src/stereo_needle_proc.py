'''
Created on Nov 6, 2020

This is a file for building image processing to segment the needle in stereo images


@author: dlezcan1

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import savgol_filter, convolve
from scipy.ndimage import convolve1d

# plotting
from matplotlib import colors as pltcolors
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

from skimage.morphology import skeletonize
from sklearn.cluster import MeanShift, estimate_bandwidth

# NURBS
from geomdl import fitting
from geomdl.visualization import VisMPL

# custom image processing
from image_processing import fit_Bspline
from BSpline1D import BSpline1D

# color HSV ranges
COLOR_HSVRANGE_RED = ( ( 0, 50, 50 ), ( 10, 255, 255 ) )
COLOR_HSVRANGE_BLUE = ( ( 110, 50, 50 ), ( 130, 255, 255 ) )
COLOR_HSVRANGE_GREEN = ( ( 40, 50, 50 ), ( 75, 255, 255 ) )
COLOR_HSVRANGE_YELLOW = ( ( 25, 50, 50 ), ( 35, 255, 255 ) )

# image size
IMAGE_SIZE = ( 768, 1024 )


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


def bin_erode( left_bin, right_bin, ksize = ( 3, 3 ) ):
    kernel = np.ones( ksize )
    left_erode = cv2.erode( left_bin, kernel )
    right_erode = cv2.erode( right_bin, kernel )
    
    return left_erode, right_erode

# bin_erode


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


def gauss_blur( left_img, right_img, ksize, sigma:tuple = ( 0, 0 ) ):
    ''' gaussian blur '''
    left_blur = cv2.GaussianBlur( left_img, ksize, sigmaX = sigma[0], sigmaY = sigma[0] )
    right_blur = cv2.GaussianBlur( right_img, ksize, sigmaX = sigma[0], sigmaY = sigma[0] )
    
    return left_blur, right_blur

# gauss_blur


def _gridproc_stereo( left_img, right_img,
                     bor_l: list = [], bor_r: list = [],
                     proc_show: bool = False ):
    ''' DEPRECATED wrapper function to segment the grid out of a stereo pair '''
    # convert to grayscale if not already
    if left_img.ndim > 2:
        left_img = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
        right_img = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )

    # if
    
    # start the image qprocessing
    left_blur, right_blur = gauss_blur( left_img, right_img, ( 5, 5 ) )
    left_thresh, right_thresh = thresh( 2.5 * left_blur, 2.5 * right_blur )
    
    left_thresh_bo = blackout_regions( left_thresh, bor_l )
    right_thresh_bo = blackout_regions( right_thresh, bor_r )
    
    left_med, right_med = median_blur( left_thresh_bo, right_thresh_bo, 5 )
    
    left_close, right_close = bin_close( left_med, right_med, ksize = ( 5, 5 ) )
    left_open, right_open = bin_open( left_close, right_close, ksize = ( 3, 3 ) )
    left_close2, right_close2 = bin_close( left_open, right_open, ksize = ( 7, 7 ) )
    left_skel, right_skel = skeleton( left_close2, right_close2 )
    
    # hough line transform
    hough_thresh = 200
    left_lines = np.squeeze( cv2.HoughLines( left_skel.astype( np.uint8 ), 2, np.pi / 180, hough_thresh ) )
    right_lines = np.squeeze( cv2.HoughLines( right_skel.astype( np.uint8 ), 2, np.pi / 180, hough_thresh ) )
    
    print( '# left lines:', len( left_lines ) )
    print( '# right lines:', len( right_lines ) )
    
    # # draw the hough lines
    left_im_lines = cv2.cvtColor( left_img, cv2.COLOR_GRAY2RGB )
    for rho, theta in left_lines:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * ( -b ) )
        y1 = int( y0 + 1000 * ( a ) )
        x2 = int( x0 - 1000 * ( -b ) )
        y2 = int( y0 - 1000 * ( a ) )

        cv2.line( left_im_lines, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
        
    # for
    
    right_im_lines = cv2.cvtColor( right_img, cv2.COLOR_GRAY2RGB )
    for rho, theta in right_lines:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * ( -b ) )
        y1 = int( y0 + 1000 * ( a ) )
        x2 = int( x0 - 1000 * ( -b ) )
        y2 = int( y0 - 1000 * ( a ) )

        cv2.line( right_im_lines, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
        
    # for
    
    # harris corner detection
#     left_centroid = cv2.cornerHarris( left_open, 2, 5, 0.04 )
#     left_centroid = cv2.dilate( left_centroid, None )
#     _, left_corners = cv2.threshold( left_centroid, 0.2 * left_centroid.max(),
#                                 255, 0 )
#     left_corners = np.int0( left_corners )
#      
#     right_centroid = cv2.cornerHarris( right_open, 2, 5, 0.04 )
#     right_centroid = cv2.dilate( right_centroid, None )
#     _, right_corners = cv2.threshold( right_centroid, 0.2 * right_centroid.max(),
#                                 255, 0 )
#     right_corners = np.int0( right_corners )
#     
#     left_crnr = cv2.cvtColor( left_img, cv2.COLOR_GRAY2RGB )
#     right_crnr = cv2.cvtColor( right_img, cv2.COLOR_GRAY2RGB )
#     left_crnr[left_corners] = [255, 0, 0]
#     right_crnr[right_corners] = [255, 0, 0]
    
    # plotting
    if proc_show:
        plt.ion()
        
        plt.figure()
        plt.imshow( imconcat( left_blur, right_blur ), cmap = 'gray' )
        plt.title( "gaussian blurring" )
        
        plt.figure()
        plt.imshow( imconcat( left_thresh, right_thresh, 150 ), cmap = 'gray' )
        plt.title( 'adaptive thresholding: after blurring' )
        
        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap = 'gray' )
        plt.title( 'region suppression: after thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_med, right_med, 150 ), cmap = 'gray' )
        plt.title( 'median: after region suppression' )

        plt.figure()
        plt.imshow( imconcat( left_close, right_close, 150 ), cmap = 'gray' )
        plt.title( 'closing: after median' )
        
        plt.figure()
        plt.imshow( imconcat( left_open, right_open, 150 ), cmap = 'gray' )
        plt.title( 'opening: after closing' )
        
        plt.figure()
        plt.imshow( imconcat( left_close2, right_close2, 150 ), cmap = 'gray' )
        plt.title( 'closing 2: after opening' )
        
        plt.figure()
        plt.imshow( imconcat( left_skel, right_skel, 150 ), cmap = 'gray' )
        plt.title( 'skeletonize: after closing 2' )
        
        plt.figure()
        plt.imshow( imconcat( left_im_lines, right_im_lines ) )
        plt.title( 'hough lines transform' )
        
    # if

# _gridproc_stereo


def gridproc_stereo( left_img, right_img,
                     bor_l: list = [], bor_r: list = [],
                     proc_show: bool = False ):
    ''' wrapper function to segment the grid out of a stereo pair '''
    # convert to grayscale if not already
    if left_img.ndim > 2:
        left_img = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
        right_img = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )

    # if
    
    # start the image processing
    left_canny, right_canny = canny( left_img, right_img, 20, 60 )
    left_bo = blackout_regions( left_canny, bor_l )
    right_bo = blackout_regions( right_canny, bor_r )
    
#====================== STANDARD HOUGH TRANSFORM  ==============================
#     # hough line transform
#     hough_thresh = 450
#     left_lines = np.squeeze( cv2.HoughLines( left_bo, 2, np.pi / 180, hough_thresh ) )
#     right_lines = np.squeeze( cv2.HoughLines( right_bo, 2, np.pi / 180, hough_thresh ) )
#     
#     print( 'Hough Transform' )
#     print( '# left lines:', len( left_lines ) )
#     print( '# right lines:', len( right_lines ) )
#     print()
#     
#     # # draw the hough lines
#     left_im_lines = cv2.cvtColor( left_img, cv2.COLOR_GRAY2RGB )
#     for rho, theta in left_lines:
#         a = np.cos( theta )
#         b = np.sin( theta )
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int( x0 + 1000 * ( -b ) )
#         y1 = int( y0 + 1000 * ( a ) )
#         x2 = int( x0 - 1000 * ( -b ) )
#         y2 = int( y0 - 1000 * ( a ) )
# 
#         cv2.line( left_im_lines, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
#         
#     # for
#     
#     right_im_lines = cv2.cvtColor( right_img, cv2.COLOR_GRAY2RGB )
#     for rho, theta in right_lines:
#         a = np.cos( theta )
#         b = np.sin( theta )
#         x0 = a * rho
#         y0 = b * rho
#         x1 = int( x0 + 1000 * ( -b ) )
#         y1 = int( y0 + 1000 * ( a ) )
#         x2 = int( x0 - 1000 * ( -b ) )
#         y2 = int( y0 - 1000 * ( a ) )
# 
#         cv2.line( right_im_lines, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
#         
#     # for
#===============================================================================

    # prob. hough line transform
    minlinelength = int( 0.8 * left_img.shape[1] )
    maxlinegap = 20
    hough_thresh = 100
    left_linesp = np.squeeze( cv2.HoughLinesP( left_bo, 1, np.pi / 180, hough_thresh, minlinelength, maxlinegap ) )
    right_linesp = np.squeeze( cv2.HoughLinesP( right_bo, 1, np.pi / 180, hough_thresh, minlinelength, maxlinegap ) )
    
    print( 'Probabilisitic Hough Transform' )
    print( "min. line length, max line gap: ", minlinelength, maxlinegap )
    print( '# left lines:', left_linesp.shape )
    print( '# right lines:', right_linesp.shape )
    print()
    
    # # Draw probabilistic hough lines 
    left_im_linesp = cv2.cvtColor( left_img, cv2.COLOR_GRAY2RGB )
    left_houghp = np.zeros( left_im_linesp.shape[0:2], dtype = np.uint8 )
    for x1, y1, x2, y2 in left_linesp:
        cv2.line( left_im_linesp, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
        cv2.line( left_houghp, ( x1, y1 ), ( x2, y2 ), ( 255, 255, 255 ), 1 )
        
    # for
    
    right_im_linesp = cv2.cvtColor( right_img, cv2.COLOR_GRAY2RGB )
    right_houghp = np.zeros( right_im_linesp.shape[0:2], dtype = np.uint8 )
    for x1, y1, x2, y2 in right_linesp:
        cv2.line( right_im_linesp, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
        cv2.line( right_houghp, ( x1, y1 ), ( x2, y2 ), ( 255, 255, 255 ), 1 )
        
    # for
    
    # hough lines on prob. hough lines image
    # hough line transform
    hough_thresh = 100
    left_lines2 = np.squeeze( cv2.HoughLines( left_houghp, 1, np.pi / 180, hough_thresh ) )
    right_lines2 = np.squeeze( cv2.HoughLines( right_houghp, 1, np.pi / 180, hough_thresh ) )
    
    print( 'Hough Transform (2)' )
    print( '# left lines:', len( left_lines2 ) )
    print( '# right lines:', len( right_lines2 ) )
    print()
    
    # # draw the hough lines
    left_im_lines2 = cv2.cvtColor( left_img, cv2.COLOR_GRAY2RGB )
    for rho, theta in left_lines2:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * ( -b ) )
        y1 = int( y0 + 1000 * ( a ) )
        x2 = int( x0 - 1000 * ( -b ) )
        y2 = int( y0 - 1000 * ( a ) )

        cv2.line( left_im_lines2, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
        
    # for
    
    right_im_lines2 = cv2.cvtColor( right_img, cv2.COLOR_GRAY2RGB )
    for rho, theta in right_lines2:
        a = np.cos( theta )
        b = np.sin( theta )
        x0 = a * rho
        y0 = b * rho
        x1 = int( x0 + 1000 * ( -b ) )
        y1 = int( y0 + 1000 * ( a ) )
        x2 = int( x0 - 1000 * ( -b ) )
        y2 = int( y0 - 1000 * ( a ) )

        cv2.line( right_im_lines2, ( x1, y1 ), ( x2, y2 ), ( 255, 0, 0 ), 2 )
        
    # for
    
    # plotting
    if proc_show:
        plt.ion()
        
        plt.figure()
        plt.imshow( imconcat( left_canny, right_canny, 150 ), cmap = 'gray' )
        plt.title( 'canny' )
        
        plt.figure()
        plt.imshow( imconcat( left_bo, right_bo, 150 ), cmap = 'gray' )
        plt.title( 'region suppression: after thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_im_linesp, right_im_linesp ) )
        plt.title( 'probabilistic hough lines transform' )
        
        plt.figure()
        plt.imshow( imconcat( left_houghp, right_houghp, 150 ), cmap = 'gray' )
        plt.title( 'prob. hough lines transform (2)' )
        
        plt.figure()
        plt.imshow( imconcat( left_im_lines2, right_im_lines2 ) )
        plt.title( 'hough lines transform: after prob. hough. lines (2)' )
        
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


def load_stereoparams_matlab( param_file: str ):
    ''' Loads the matlab stereo parameter file created from a struct '''
    
    mat = sio.loadmat( param_file )
    
    stereo_params = {}
    
    keys = ['cameraMatrix1', 'cameraMatrix2', 'distCoeffs1',
            'distCoeffs2', 'R1', 'tvecs1', 'R2', 'tvecs2',
            'R', 't', 'F', 'E', 'units']

    # load stereo parameters    
    for key in keys:
        if key == 'units':
            stereo_params[key] = mat[key][0]
            
        elif ( key == 'R1' ) or ( key == 'R2' ):
            stereo_params[key + '_ext'] = mat[key]
            
        else:
            stereo_params[key] = mat[key]
        
    # for
    
    # projection matrices
    R1, R2, P1, P2, Q, *_ = cv2.stereoRectify( stereo_params['cameraMatrix1'], stereo_params['distCoeffs1'],
                                              stereo_params['cameraMatrix2'], stereo_params['distCoeffs2'],
                                              ( 768, 1024 ), stereo_params['R'], stereo_params['t'] )
    R = stereo_params['R']
    t = stereo_params['t']
    H = np.vstack( ( np.hstack( ( R, t.reshape( 3, 1 ) ) ) , [0, 0, 0, 1] ) )
   
    stereo_params['R1'] = R1
    stereo_params['R2'] = R2
    stereo_params['P1'] = stereo_params['cameraMatrix1'] @ np.eye( 3, 4 )
    stereo_params['P2'] = stereo_params['cameraMatrix2'] @ H[:-1]
    stereo_params['Q'] = Q
    
    return stereo_params

# load_stereoparams_matlab
    

def imconcat( left_im, right_im, pad_val = 0, pad_size = 20 ):
    ''' wrapper for concatenating images'''
    
    if left_im.ndim == 2:
        pad_left_im = np.pad( left_im, ( ( 0, 0 ), ( 0, pad_size ) ), constant_values = pad_val )
    
    elif left_im.ndim == 3:
        if np.ndim( pad_val ) > 0:  # color img
            pad_val = ( ( pad_val, pad_val ), ( pad_val, pad_val ), ( 0, 0 ) )
        
        # if
        
        pad_left_im = np.pad( left_im, ( ( 0, 0 ), ( 0, pad_size ), ( 0, 0 ) ), constant_values = pad_val )
    
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
                         roi_l:tuple = (), roi_r:tuple = (),
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
    
    left_roi = roi( left_thresh, roi_l, full = True )
    right_roi = roi( right_thresh, roi_r, full = True )
    
    left_thresh_bo = blackout_regions( left_roi, bor_l )
    right_thresh_bo = blackout_regions( right_roi, bor_r )
    
    left_tmed, right_tmed = median_blur( left_thresh_bo, right_thresh_bo, ksize = 5 )
    
    left_open, right_open = bin_open( left_tmed, right_tmed, ksize = ( 5, 5 ) )
    
    left_close, right_close = bin_close( left_open, right_open, ksize = ( 7, 7 ) )
    
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
        plt.imshow( imconcat( left_roi, right_roi, 150 ), cmap = 'gray' )
        plt.title( 'roi: after thresholding' )
        
        plt.figure()
        plt.imshow( imconcat( left_thresh_bo, right_thresh_bo, 150 ), cmap = 'gray' )
        plt.title( 'region suppression: roi' )
        
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


def roi( img, roi, full:bool = True ):
    ''' return region of interest 
    
        @param roi: [tuple of top-left point, tuple of bottom-right point]
        
        @return: subimage of the within the roi
    '''
    
    if len( roi ) == 0:
        return img
    
    # if
    
    tl_i, tl_j = roi[0]
    br_i, br_j = roi[1]
    
    if full:
        img_roi = img.copy()
        
        # zero-out value
        zval = 0 if img.ndim == 2 else np.array( [0, 0, 0] )
        
        # zero out values
        img_roi [:tl_i,:] = zval
        img_roi [br_i:,:] = zval
        
        img_roi [:,:tl_j] = zval
        img_roi [:, br_j:] = zval
        
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


def stereo_disparity( left_gray, right_gray, stereo_params: dict ):
    ''' stereo distparity mapping '''
    # parameters
    win_size = 5
    
    left_gauss, right_gauss = gauss_blur( left_gray, right_gray, ksize = ( 1, 1 ) )
    stereo = cv2.StereoSGBM_create( numDisparities = 64,
                                    blockSize = win_size,
                                    speckleRange = 2, speckleWindowSize = 5,
                                    P1 = 8 * 3 * win_size ** 2,
                                    P2 = 20 * 3 * win_size ** 2 )
    disparity = stereo.compute( left_gauss, right_gauss )
    
    return disparity
    
# stereo_disparity


def stereomatch_needle( left_conts, right_conts, start_location = "tip", col:int = 1 ):
    ''' stereo matching needle arclength points for the needle
        
        
        Args:
            (left/right)_conts: a nx2 array of pixel coordinates
                                for the contours in the (left/right) image
            
            start_location (Default: "tip"): a string of where to start counting.
                                             tip is only implemented.
                                             
            col (int = 1): the column to begin matching by
    
     '''
    # squeeze dimensions just in case
    left_conts = np.squeeze( left_conts )
    right_conts = np.squeeze( right_conts )
    
    # remove duplicate rows
    left_conts = np.unique( left_conts, axis = 0 )
    right_conts = np.unique( right_conts, axis = 0 )
    
    # find the minimum number of points to match
    n = min( left_conts.shape[0], right_conts.shape[0] )
    
    if start_location.lower() == "tip":
        left_idx = np.argsort( left_conts[:, col] )[-n:]
        right_idx = np.argsort( right_conts[:, col] )[-n:]
        
        left_matches = left_conts[left_idx]
        right_matches = right_conts[right_idx]
        
    # if
    
    else:
        raise ValueError( f"start_location = {start_location} not valid." )
    
    # else
    
    return left_matches, right_matches

# stereomatch_needle


def triangulate_points( pts_l, pts_r, stereo_params: dict, distorted:bool = False ):
    ''' function to perform 3-D reconstruction of the pts in left and right images.
    
        @param pts_(l/r): the left/right image points to triangulate of size [Nx2]
        @param stereo_params: dict of the stereo parameters
        @param distorted (bool, Default=True): whether to undistort the pts in each image
        
        DO NOT USE 'distorted'! This causes major errors @ the moment.
        
        @return: [Nx3] world frame points
        
    '''
    # load in stereo parameters, camera matrices and distortion coefficients
    # - camera matrices
    Kl = stereo_params['cameraMatrix1']
    distl = stereo_params['distCoeffs1']

    Kr = stereo_params['cameraMatrix2']
    distr = stereo_params['distCoeffs2']
    
    # - stereo parameters
    R = stereo_params['R']
    t = stereo_params['t']

    # convert to float types
    pts_l = np.float64( pts_l )
    pts_r = np.float64( pts_r )
    
    # undistort the points if needed
    if distorted:
        pts_l, pts_r = undistort_points( pts_l, pts_r, stereo_params )
        
        # get undistorted camera params
        Kl = stereo_params['cameraMatrix1_new'] 
        Kr = stereo_params['cameraMatrix2_new']

    # if
    
    # calculate projection matrices
    Pl = Kl @ np.eye( 3, 4 )
    H = np.vstack( ( np.hstack( ( R, t.reshape( 3, 1 ) ) ), [0, 0, 0, 1] ) )
    Pr = Kr @ H[0:3]
    
    # - transpose to [2 x N]
    pts_l = pts_l.T
    pts_r = pts_r.T
    
    # perform triangulation of the points
    pts_3d = cv2.triangulatePoints( Pl, Pr, pts_l, pts_r )
    pts_3d /= pts_3d[3]  # normalize the triangulation points

    return pts_3d[:-1]

# triangulate


def thresh( left_img, right_img, thresh = 'adapt' ):
    ''' image thresholding'''
    
    if thresh.lower() == 'adapt':
        left_thresh = cv2.adaptiveThreshold( left_img.astype( np.uint8 ), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 13, 4 )
        right_thresh = cv2.adaptiveThreshold( right_img.astype( np.uint8 ), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 13, 4 )
    
    # if
    
    elif isinstance( thresh, ( float, int ) ):
        _, left_thresh = cv2.threshold( left_img, thresh, 255, cv2.THRESH_BINARY_INV )
        _, right_thresh = cv2.threshold( right_img, thresh, 255, cv2.THRESH_BINARY_INV )
        
    # elif
    
    else:
        raise ValueError( f"thresh: {thresh} is not a valid thresholding." )
    
    return left_thresh, right_thresh

# thresh


def undistort( left_img, right_img, stereo_params:dict ):
    ''' stereo wrapper to undistort '''
    # load in camera matrices and distortion coefficients
    Kl = stereo_params['cameraMatrix1']
    distl = stereo_params['distCoeffs1']

    Kr = stereo_params['cameraMatrix2']
    distr = stereo_params['distCoeffs2']
    
    # undistort/recitfy the images
    hgtl, wdtl = left_img.shape[:2]
    Kl_new, roi = cv2.getOptimalNewCameraMatrix( Kl, distl, ( wdtl, hgtl ), 1, ( wdtl, hgtl ) )
    xl, yl, wl, hl = roi
    left_img_rect = cv2.undistort( left_img, Kl, distl, None, Kl_new )[yl:yl + hl, xl:xl + wl]
    
    hgtr, wdtr = right_img.shape[:2]
    Kr_new, roi = cv2.getOptimalNewCameraMatrix( Kr, distr, ( wdtr, hgtr ), 1, ( wdtr, hgtr ) )
    xr, yr, wr, hr = roi
    right_img_rect = cv2.undistort( right_img, Kr, distr, None, Kr_new )[yr:yr + hr, xr:xr + wr]
    
    return left_img_rect, right_img_rect
    
# undistort


def undistort_points( pts_l, pts_r, stereo_params:dict ):
    ''' wrapper for undistorting points
        
        pts is of shape [N x 2]
        
    '''
    # load in camera matrices and distortion coefficients
    Kl = stereo_params['cameraMatrix1']
    distl = stereo_params['distCoeffs1']

    Kr = stereo_params['cameraMatrix2']
    distr = stereo_params['distCoeffs2']
    
    # calculate optimal camera matrix
    Kl_new, _ = cv2.getOptimalNewCameraMatrix( Kl, distl, IMAGE_SIZE, 1, IMAGE_SIZE )
    Kr_new, _ = cv2.getOptimalNewCameraMatrix( Kr, distr, IMAGE_SIZE, 1, IMAGE_SIZE )
    
    stereo_params['cameraMatrix1_new'] = Kl_new
    stereo_params['cameraMatrix2_new'] = Kr_new
    
    # undistort the image points
    pts_l_undist = cv2.undistortPoints( np.expand_dims( pts_l, 1 ), Kl, distl,
                                        None, Kl_new ).squeeze()
    pts_r_undist = cv2.undistortPoints( np.expand_dims( pts_r, 1 ), Kr, distr,
                                        None, Kr_new ).squeeze()
    
    return pts_l_undist, pts_r_undist
    
# undistort_points


def main_dbg():
    # directory settings
    stereo_dir = "../Test Images/stereo_needle/"
    needle_dir = stereo_dir + "needle_examples/"
    grid_dir = stereo_dir + "grid_only/"
    
    # the left and right image to test
    num = 5
    left_fimg = needle_dir + f"left-{num:04d}.png"
    right_fimg = needle_dir + f"right-{num:04d}.png"
    
    # load matlab stereo calibration parameters
    stereo_param_dir = "../Stereo_Camera_Calibration_10-23-2020"
    stereo_param_file = stereo_param_dir + "/calibrationSession_params-error_opencv-struct.mat"
    stereo_params = load_stereoparams_matlab( stereo_param_file )
    
#     # read in the images and convert to grayscale
#     left_img = cv2.imread( left_fimg, cv2.IMREAD_COLOR )
#     right_img = cv2.imread( right_fimg, cv2.IMREAD_COLOR )
#     left_gray = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
#     right_gray = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )
     
#     # test undistort function ( GOOD )
#     left_rect, right_rect = undistort( left_img, right_img, stereo_params )
#     test_arr = np.zeros( ( 3, 2 ) )
#     undist_pts = undistort_points( test_arr, test_arr, stereo_params )
#     print( np.hstack( undist_pts ) )  
#     print()  
    
    # test point triangulation ( GOOD )
    world_points = np.random.randn( 3, 5 )
    world_pointsh = np.vstack( ( world_points, np.ones( ( 1, world_points.shape[1] ) ) ) )
    Pl = stereo_params['P1']
    Pr = stereo_params['P2']
    pts_l = Pl @ world_pointsh
    pts_l = ( pts_l / pts_l[-1] ).T[:,:-1]
    pts_r = Pr @ world_pointsh
    pts_r = ( pts_r / pts_r[-1] ).T[:,:-1]
    
    print( 'pts shape (l,r):', pts_l.shape, pts_r.shape )
    tri_pts = triangulate_points( pts_l, pts_r, stereo_params, distorted = False )
    print( 'World points' )
    print( world_points )
    print()
    print( 'triangulated points' )
    print( tri_pts )
    print()
    
#     # plotting / showing image results
#     plt.ion()
#     
#     plt.figure()
#     plt.imshow( imconcat( left_img, right_img, [0, 0, 255] ) )
#     plt.title( 'original image' )
#     
#     plt.figure()
#     plt.imshow( imconcat( left_rect, right_rect, [0, 0, 255] ) )
#     plt.title( 'undistorted image' )
#     
#     # close on enter
#     plt.show()
#     while True:
#         if plt.waitforbuttonpress( 0 ):
#             break
#         
#     # while
    
    plt.close( 'all' )
    
# main_dbg


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
    
    # color segmentation ( red for border )
    lmask, rmask, lcolor2, rcolor2 = color_segmentation( left_img2, right_img2, "red" )
    lcolor = cv2.cvtColor( lcolor2, cv2.COLOR_BGR2RGB )
    rcolor = cv2.cvtColor( rcolor2, cv2.COLOR_BGR2RGB )
    
    # plotting
    plt.ion()
    
#     plt.figure()
#     plt.imshow( imconcat( left_img, right_img ) )
#     plt.title( 'Original images' )
    
#     plt.figure()
#     plt.imshow( imconcat( lcolor, rcolor ) )
#     plt.title( 'masked red color' )

    # find the grid
    gridproc_stereo( left_gray, right_gray, proc_show = True )
    
    # close on enter
    plt.show()
    while True:
        if plt.waitforbuttonpress( 0 ):
            break
        
    # while
    
    plt.close( 'all' )

# main_gridproc


def main_needleproc( file_num, img_dir, save_dir = None, proc_show = False, res_show = False ):
    ''' main method for segmenting the needle centerline in stereo images'''
    # load matlab stereo calibration parameters
    stereo_param_dir = "../Stereo_Camera_Calibration_10-23-2020"
    stereo_param_file = stereo_param_dir + "/calibrationSession_params-error_opencv-struct.mat"
    stereo_params = load_stereoparams_matlab( stereo_param_file )
    
    # the left and right image to test
    left_fimg = img_dir + f"left-{file_num:04d}.png"
    right_fimg = img_dir + f"right-{file_num:04d}.png"
    
    left_img = cv2.imread( left_fimg, cv2.IMREAD_COLOR )
    right_img = cv2.imread( right_fimg, cv2.IMREAD_COLOR )
    left_gray = cv2.cvtColor( left_img, cv2.COLOR_BGR2GRAY )
    right_gray = cv2.cvtColor( right_img, cv2.COLOR_BGR2GRAY )
    
    print( 'Image shape:', left_gray.shape, end = '\n\n' + 80 * '=' + '\n\n' )
    
    # blackout regions
    bor_l = [( left_gray.shape[0] - 100, 0 ), left_gray.shape]
    bor_r = bor_l
    
    # regions of interest
    roi_l = ( ( 70, 80 ), ( 500, 915 ) )
    roi_r = ( ( 70, 55 ), ( 500, -1 ) )
    
    # needle image processing
    print( 'Processing stereo pair images...' )
    left_skel, right_skel, conts_l, conts_r = needleproc_stereo( left_img, right_img,
                                                                 bor_l = [bor_l], bor_r = [bor_r],
                                                                 roi_l = roi_l, roi_r = roi_r,
                                                                 proc_show = proc_show )
    print( 'Stereo pair processed. Contours extracted.', end = '\n\n' + 80 * '=' + '\n\n' )

    left_cont = left_img.copy()
    left_cont = cv2.drawContours( left_cont, conts_l, 0, ( 255, 0, 0 ), 12 )
    
    right_cont = right_img.copy()
    right_cont = cv2.drawContours( right_cont, conts_r, 0, ( 255, 0, 0 ), 12 )
    
    # matching contours
    print( 'Performing stereo triangulation...' )
    cont_l_match, cont_r_match = stereomatch_needle( conts_l[0], conts_r[0], start_location = 'tip', col = 1 )
    
    left_match = left_cont.copy()
    cv2.drawContours( left_match, [np.vstack( ( cont_l_match, np.flip( cont_l_match, 0 ) ) )], 0, ( 0, 255, 0 ), 4 )
    
    right_match = right_cont.copy()
    cv2.drawContours( right_match, [np.vstack( ( cont_r_match, np.flip( cont_r_match, 0 ) ) )], 0, ( 0, 255, 0 ), 4 )
    
    # draw lines from matching points
    plot_pt_freq = int( 0.1 * len( cont_l_match ) )
    pad_width = 20
    lr_match = imconcat( left_match, right_match, pad_val = [0, 0, 255], pad_size = pad_width )
    for ( x_l, y_l ), ( x_r, y_r ) in zip( cont_l_match[::plot_pt_freq ], cont_r_match[::plot_pt_freq ] ):
        cv2.line( lr_match, ( x_l, y_l ), ( x_r + pad_width + right_match.shape[1], y_r ), [255, 0, 255], 2 )
        
    # for
    
    # perform triangulation on points
    cont_match_3d = triangulate_points( cont_l_match, cont_r_match, stereo_params, distorted = True )
    
    # - smooth 3-D points
    print( 'Smoothing 3-D stereo points and fitting 3-D NURBS...' )
    win_size = 55
    cont_match_3d_sg = savgol_filter( cont_match_3d, win_size, 1, deriv = 0 )  
        
    # - NURBS fitting 
    nurbs = fitting.approximate_curve( cont_match_3d_sg.T.tolist(), degree = 2, ctrlpts_size = 35 )
    nurbs.delta = 0.005
    nurbs.vis = VisMPL.VisCurve3D()
    print( 'Smoothing and NURBS fit.', end = '\n\n' + 80 * '=' + '\n\n' )
    
    # test disparity mapping
    disparity = stereo_disparity( left_img, right_img, stereo_params )
    
    # show results
    if res_show:
        print( 'Plotting...' )
        plt.ion()
        
        plt.figure()
        plt.imshow( imconcat( left_cont, right_cont, pad_val = [0, 0, 255] ) )
        plt.title( 'Contours of needle' )
        
        plt.figure()
        plt.imshow( lr_match )
        plt.title( 'matching contour points' )
        
        extras = [
                  dict( points = cont_match_3d.T.tolist(),
                        name = 'triangulation',
                        color = 'red',
                        size = 1 ),
                  dict( points = cont_match_3d_sg.T.tolist(),
                       name = 'savgol_filter',
                       color = 'green',
                       size = 1 )
                  ]        
        nurbs.render( extras = extras )
        ax = plt.gca()
        axisEqual3D( ax )
        
#==================== OLD 3-D PLOTTING =========================================
#         f3d = plt.figure()
#         ax = fig3d.add_subplot(111, projection='3d')
#         ax.plot( cont_match_3d[0], cont_match_3d[1], cont_match_3d[2], '.' , label = 'triangulation' )
#         ax.plot( cont_match_3d_sg[0], cont_match_3d_sg[1], cont_match_3d_sg[2], '-' , label = 'savgol_filter' )
# #         ax.plot( cont_match_3d_mvavg[0], cont_match_3d_mvavg[1], cont_match_3d_mvavg[2], '-' , label='moving average')
#         plt.legend( [ 'nurbs', 'triangulation', 'savgol_filter'] )
#         axisEqual3D( ax )
#         plt.title( '3-D needle reconstruction' )
#===============================================================================
        
        plt.figure()
        plt.imshow( disparity, cmap = 'gray' )
        plt.title( 'stereo disparity map' )
        
        # close on enter
        print( 'Press any key on the last figure to close all windows.' )
        plt.show()
        while True:
            try:
                if plt.waitforbuttonpress( 0 ):
                    break
                
                # if
            # try
            
            except:
                break
            
            # except    
        # while
        
        print( 'Closing all windows...' )
        plt.close( 'all' )
        print( 'Plotting finished.', end = '\n\n' + 80 * '=' + '\n\n' )
        
    # if
    
    # save the processed images
    if save_dir:
        print( 'Saving figures and files...' )
        save_fbase = save_dir + f"left-right-{file_num:04d}" + "_{:s}.png"
        save_fbase_txt = save_dir + f"left-right-{file_num:04d}" + "_{:s}.txt"
        # - skeletons
        plt.imsave( save_fbase.format( 'skel' ), imconcat( left_skel, right_skel, 150 ),
                    cmap = 'gray' )
        print( 'Saved figure:', save_fbase.format( 'skel' ) )
        
        # - contours
        plt.imsave( save_fbase.format( 'cont' ), imconcat( left_cont, right_cont, pad_val = [0, 0, 255] ) )
        print( 'Saved figure:', save_fbase.format( 'cont' ) )
        
        # - matching contours
        plt.imsave( save_fbase.format( 'cont-match' ), lr_match )
        print( 'Saved Figure:', save_fbase.format( 'cont-match' ) )
        
        # - 3D reconstruction
        extras = [
                  dict( points = cont_match_3d.T.tolist(),
                        name = 'triangulation',
                        color = 'red',
                        size = 1 ),
                  dict( points = cont_match_3d_sg.T.tolist(),
                       name = 'savgol_filter',
                       color = 'green',
                       size = 1 )
                  ]    
        nurbs.render( plot = False, filename = save_fbase.format( '3d-reconstruction' ), extras = extras )
        ax = plt.gca()
        axisEqual3D( ax )
        print( 'Saved Figure:', save_fbase.format( '3d-reconstruction' ) )
        
        np.savetxt( save_fbase_txt.format( 'cont-match' ), np.hstack( ( cont_l_match, cont_r_match ) ) )
        print( 'Saved file:', save_fbase_txt.format( 'cont-match' ) )
        
        np.savetxt( save_fbase_txt.format( 'cont-match_3d' ), cont_match_3d )
        print( 'Saved file:', save_fbase_txt.format( 'cont-match_3d' ) )
        
        plt.close()
        print( 'Finished saving files and figures.', end = '\n\n' + 80 * '=' + '\n\n' )
        
    # if
    
    return cont_l_match, cont_r_match

# main_needleproc


def axisEqual3D( ax ):
    ''' taken from online '''
    extents = np.array( [getattr( ax, 'get_{}lim'.format( dim ) )() for dim in 'xyz'] )
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean( extents, axis = 1 )
    maxsize = max( abs( sz ) )
    r = maxsize / 2
    for ctr, dim in zip( centers, 'xyz' ):
        getattr( ax, 'set_{}lim'.format( dim ) )( ctr - r, ctr + r )
        
# axisEqual3D


if __name__ == '__main__':
    # directory settings
    stereo_dir = "../Test Images/stereo_needle/"
    needle_dir = stereo_dir + "needle_examples/"
    grid_dir = stereo_dir + "grid_only/"
    
#     # load matlab stereo calibration parameters
#     stereo_param_dir = "../Stereo_Camera_Calibration_10-23-2020"
#     stereo_param_file = stereo_param_dir + "/calibrationSession_params-error_opencv-struct.mat"
#     stereo_params = load_stereoparams_matlab( stereo_param_file )
    
    # iteratre through the current gathered images
    for i in range( -1 ):
        try:
            main_needleproc( i, needle_dir, needle_dir, proc_show = False, res_show = False )
            
        except:
            print( 'passing:', i )
            
        print()
        
    # for
    
    # testing functions
    main_needleproc( 5, needle_dir, None, proc_show = False, res_show = True )
#     main_gridproc( 2, grid_dir, grid_dir )
#     main_dbg()
    
    print( 'Program complete.' )

# if


'''
Created on Dec 6, 2019

@author: Dimitri Lezcano

@summary: This script is intended for the use of performing the data processing
          incorporating the image and FBG data when needle calibration is 
          performed.
'''

import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import glob, re
import time  # for debugging purposes
import warnings
import image_processing as imgp
from PIL.ImageOps import crop
from image_processing import get_centerline
import cv2
from matplotlib.testing.jpl_units import sec

TIME_FMT = "%H-%M-%S.%f"
TIME_FIX = timedelta( hours = 3 )  # the internal fix for the time

PIX_PER_MM = 8.498439767625596
# CROP_AREA = ( 84, 250, 1280, 715 )
CROP_AREA = ( 32, 425, 1180, 580 )
BO_REGIONS = [( 0, 70, 165, -1 ), ( 0, 0, -1, 19 )]
BO_REGIONS.append( ( 0, 0, -1, 30 ) )
BO_REGIONS.append( ( 0, 60, 20, -1 ) )


def fix_fbgData( filename: str ):
    """ function to fix the fbg data formatting (remove unnecessary line breaks)"""

    timestamps = np.empty( 0 )
    fbgdata = np.empty( 0 )
    
    with open( filename, 'r' ) as file:
        lines = file.read().split( ":" )
        
    # with
    
    new_file = filename.split( '/' )
    new_file[-1] = "fixed_" + new_file[-1]
    new_file = '/'.join( new_file )
    
    with open( new_file, 'w+' ) as writestream:
        for i, line in enumerate( lines ):
#             print( 100 * i / len( lines ), '%' )
            
            d = line.split( ' ' )
            if len( d ) == 1 and i < len( lines ) - 1:  # not at the end
                timestamps = np.append( timestamps, line + ": " )
                
            elif len( d ) == 1 and i >= len( lines ) - 1:  # at the end
                fbgdata = np.append( fbgdata, line.replace( '\n', '' ) )
            
            else:
                ts = line.split( '\n' )[-1] + ':'
                data = line.split( '\n' )[:-1]
                data = " ".join( data ).replace( '\n', '' )
#                 data = " ".join( data.split( '\n' ) )
                timestamps = np.append( timestamps, ts )
                fbgdata = np.append( fbgdata, data )
                
            # else
            
            if len( timestamps ) > 0 and len( fbgdata ) > 0:
                ts = timestamps[0]
                timestamps = timestamps[1:]
                d = fbgdata[0]
                fbgdata = fbgdata[1:]
                
                writestream.write( ts + d + '\n' )
                
            # if
            
    # with

    print( "Wrote file:", new_file )
#     with open( new_file, 'w+' ) as file:
#             for ts, d in zip( timestamps, fbgdata ):
#                 file.write( ts + d + '\n' );
#                 
#     # with
        
# fix_fbgData
                
    
def read_fbgData( filename: str , num_active_areas: int, lines: list = [-1] ):
    """ Function to read in the FBG data
    
        @param filename: str, the input fbg data file
        
        @param num_active_areas: int, representing the number of active areas
        
        @param lines: list of line numbers that would like to be read in. 
                        (default = [-1], indicating all lines)
        
        @return: (timestamps, fbgdata)
                 timestamps: numpy array of timestamps corresponding row-wise
                                to fbgdata
                 fbgdata:    numpy array of fbg readings where the rows are 
                                 the time per entry
                                 
    """
    global TIME_FMT, TIME_FIX
    
    max_lines = max( lines )
    timestamps = np.empty( 0 )
    fbgdata = np.empty( ( 0, 3 * num_active_areas ) )
    
    with open( filename, 'r' ) as file:
        for i, line in enumerate( file ):
            
            # read in desired lines
            if lines != [-1]:
                
                if i > max_lines:  # passed the maximum lines, stop reading
                    break
                
                elif i not in lines:  # not a line we want to read
                    continue
            
            # if
            
            ts, datastring = line.split( ':' )
            ts = datetime.strptime( ts, TIME_FMT ) + TIME_FIX
            timestamps = np.append( timestamps, ts )
            data = np.fromstring( datastring, sep = ',' , dtype = float )
            
            if len( data ) == 3 * num_active_areas:  # all active areas measured
                fbgdata = np.vstack( ( fbgdata, data ) )
                
            else:  # too many or not enough sensor readings
                fbgdata = np.vstack( ( fbgdata, -1 * np.ones( 3 * num_active_areas ) ) )
                
        # for
    # with
        
    return timestamps, fbgdata
            
# read_fbgData


def read_needleparam( filename: str ):
    """ Function to read the needle parameters file
    
        @param filename: str, representing the needle parameter filename
        
        
        @return (needle_length, # active_areas, active_areas)
                    needle_length  = the length of the needle (in mm)
                    # active_areas = number of active areas
                    active_areas   = numpy 1-d array of the distances of active
                                         areas from the tip (in mm)
                                    
    """
    with open( filename, 'r' ) as file:
        lines = file.read().split( '\n' )
        _, needle_length, num_active_areas = lines[0].split( ',' )
        
        needle_length = float( needle_length )
        num_active_areas = int( num_active_areas )
        
        active_areas = np.fromstring( lines[1], sep = ',', dtype = float )
        
    # with
    
    return needle_length, num_active_areas, active_areas

# read_needleparam


def get_FBGdata_windows( lutimes: np.ndarray, dt: float, timestamps: np.ndarray, max_match: int = 100 ):
    """ Function to find the windowed timedata from the gathered FBGdata.
    
        @param lutime: list of items to look up
        
        @param dt:     float, the time window length in seconds.
        
        @param timestamps: numpy 1-D array of timestamps
        
        @param max_match: (optional, default = 100) number of maximum matches
                
        @return 1-D array -> indices matching to the timestamps matched
                
    """

    retval = -1 * np.ones( ( len( lutimes ), max_match ) )  # instantiate no matches
    
    dT = timedelta( seconds = dt )
    
    for kk, lut in enumerate( lutimes ):
        idxs = np.logical_and( timestamps >= lut - dT, timestamps <= lut + dT )
        idxs = np.argwhere( idxs ).reshape( -1 )
        retval[kk, :len( idxs )] = idxs[:max_match]
        
    # for
    
    return retval

# get_FBGdata_window


def get_curvature_image ( filename: str, active_areas: np.ndarray, needle_length: float,
                          display: bool = False ):
    """ Method to get curvature @ the active areas & output to data file."""
    global PIX_PER_MM, CROP_AREA, BO_REGIONS
    
    img, gray_img = imgp.load_image( filename )
    
    gray_img = imgp.saturate_img( gray_img, 1.75, 15 )
    crop_img = imgp.set_ROI_box( gray_img, CROP_AREA )
    print( BO_REGIONS )
    canny_img = imgp.canny_edge_detection( crop_img, display, BO_REGIONS )
    
    skeleton = imgp.get_centerline ( canny_img )
    if display:
        cv2.imshow( "Skeleton", skeleton )
    
    # if
    
    if display:
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
    
    # if
    
    poly, x = imgp.fit_polynomial( skeleton, 12 )
    x = np.sort( x )  # sort the x's  (just in case)
    
    plt.figure()
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    imgp.plot_func_image( crop_img, poly, x )
    plt.show()
    
    k = input( "Does this plot look ok to proceed? (y/n) " )
    if k.lower() == 'n':
        print( "Not continuing with the curvature analysis." )
        return -1
    # if

    
    print( "Continuing with the curvature analysis." )
    
    x0 = x.max()  # start integrating from the tip
    lb = x.min()
    ub = x.max()
    x_active = imgp.find_active_areas( x0, poly, active_areas, PIX_PER_MM, lb, ub )

    curvature = imgp.fit_circle_curvature( poly, x, x_active, 20 * PIX_PER_MM )
    x_active = np.array( x_active ).reshape( -1 )
    
    outfile = '.'.join( filename.split( '.' )[:-1] ) + '.txt'
    outfile = outfile.replace( 'mono', 'curvature_mono' )
    
    print( "Completed curvature analysis." )
    
    with open( outfile, 'w+' ) as writestream:
        writestream.write( "Curvature of the needle @ the active areas.\n" )
        
        writestream.write( f"Pixels/mm: {PIX_PER_MM}\n" )
        
        writestream.write( "Active areas (mm): " )
        msg = np.array2string( active_areas, separator = ', ',
                                  max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
        
        writestream.write( "Curvatures (1/mm): " )
        msg = np.array2string( curvature, separator = ', ',
                                 precision = 10, max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
        
        writestream.write( "Polynomial coefficients: " )
        msg = np.array2string( poly.coef , separator = ', ',
                                 precision = 10, max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
        
        writestream.write( "x range of center line (px): " )
        msg = f"[ {x.min()}, {x.max()} ]"
        writestream.write( msg + '\n' )
        
        writestream.write( "x-values of active areas (px): " )
        msg = np.array2string( x_active, separator = ', ',
                                 precision = 10, max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
    
    # with
    
    print( "Wrote outfile:", outfile )
    return 0
    
# get_curvature_image


def main():
    directory = "../FBG_Needle_Calibration_Data/needle_1/"
    
    needleparam = directory + "needle_params.csv"
    num_actives, length, active_areas = read_needleparam( needleparam )
    
    directory += "12-09-19_14-01/"
    
    imgfiles = glob.glob( directory + "monofbg*.jpg" )
    
    img_patt = r"monofbg_12-09-2019_([0-9][0-9])-([0-9][0-9])-([0-9][0-9]).([0-9]+).jpg"
    
    imgfiles = ["../FBG_Needle_Calibration_Data/needle_1/12-09-19_14-01\monofbg_12-09-2019_14-06-03.085297826.jpg",
                "../FBG_Needle_Calibration_Data/needle_1/12-09-19_14-01\monofbg_12-09-2019_14-04-31.217293380.jpg",
                ]
    for imgf in imgfiles:
#         hr, mn, sec, ns = re.search( img_patt, imgf ).groups()
        print( "Processing file:" , imgf )
#         str_ts = f"{hr}:{mn}:{sec}.{ns[0:6]}"
#         ts = datetime.strptime( str_ts, "%H:%M:%S.%f" )
#         print( ts )
        
        get_curvature_image( imgf, active_areas, length, True )
        print()
        
    # for
        
# main


if __name__ == '__main__':
    main()
#     directory = "../FBG_Needle_Calibaration_Data/needle_1/"
#     img_file = "12-09-19_12-29/mono_0001.jpg"
#     a = np.empty( 0 )
#     
#     get_curvature_image( directory + img_file, a, 0 , True )
    
    print( "Program has terminated." )
    
# if
    

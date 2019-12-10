'''
Created on Dec 6, 2019

@author: Dimitri Lezcano

@summary: This script is intended for the use of performing the data processing
          incorporating the image and FBG data when needle calibration is 
          performed.
'''

import numpy as np
from datetime import datetime, timedelta
import glob
import time  # for debugging purposes
import warnings

TIME_FMT = "%H-%M-%S.%f"


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
                
    
def read_fbgData( filename: str , num_active_areas: int ):
    """ Function to read in the FBG data
    
        @param filename: str, the input fbg data file
        
        @param num_active_areas: int, representing the number of active areas
        
        @return: (timestamps, fbgdata)
                 timestamps: numpy array of timestamps corresponding row-wise
                                to fbgdata
                 fbgdata:    numpy array of fbg readings where the rows are 
                                 the time per entry
                                 
    """
    global TIME_FMT
    
    timestamps = np.empty( 0 )
    fbgdata = np.empty( ( 0, 3 * num_active_areas ) )
    
    with open( filename, 'r' ) as file:
        for i, line in enumerate( file ):
            
            if i >= 5000:
                break
            
            ts, datastring = line.split( ':' )
            ts = datetime.strptime( ts, TIME_FMT )
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


def get_FBGdata_windows( lutimes: np.ndarray, dt: float, timestamps: np.ndarray ):
    """ Function to find the windowed timedata from the gathered FBGdata.
    
        @param lutime: list of items to look up
        
        @param dt:     float, the time window length in seconds.
        
        @param timestamps: numpy 1-D array of timestamps
                
        @return 1-D array -> indices matching to the timestamps matched
                
                
    """
    lutimes = sorted( lutimes )
    timestamps = sorted( timestamps )
    MAX_MATCHES = 100
    retval = -1 * np.ones( ( len( lutimes ), MAX_MATCHES ) )  # instantiate no matches
    
    dT = timedelta( seconds = dt )
    
    ii = 0  # pointer in lutimes
    for kk, lut in enumerate( lutimes ):
        jj = 0  # pointer in retval
        
        while ii >= 0 and lut > timestamps[ii] :
            ii -= 1  # back it up
            
        # check to see if you ran past the timestamps
        if ii < 0:
            warnings.warn( f"look-up time: {lut} is not in this set (less than). Passing over" )
            ii = 0
            continue
        
        while ii < len( lutimes ) and lut <= timestamps[ii]  :
            if timestamps[ii] >= lut - dT and timestamps[ii] <= lut + dT:  # lut - dt <= lutimes <= lut + dt
                retval[kk, jj] = ii
                jj += 1
                
                if jj >= MAX_MATCHES:  # don't overshoot
                    break
            
            # if
            
            ii += 1  # increment to next greater time
            
        # while
                
        if ii >= len( timestamps ) + 1:
            warnings.warn( f"look-up time: {lut} is not in this set (greater than). Passing over" )
            ii = len( timestamps ) - 1
            continue
            
        # if
        
        ii -= 1  # go back to the last element
    # for
    
    return retval
            
# get_FBGdata_window


def get_window_numpy( lutimes: np.ndarray, dt: float, timestamps: np.ndarray ):
    MAX_MATCHES = 100
    retval = -1 * np.ones( ( len( lutimes ), MAX_MATCHES ) )  # instantiate no matches
    
    dT = timedelta( seconds = dt )
    
    for kk, lut in enumerate( lutimes ):
        idxs = np.logical_and( timestamps >= lut - dT, timestamps <= lut + dT )
        idxs = np.argwhere( idxs ).reshape( -1 )
        retval[kk, :len( idxs )] = idxs[:MAX_MATCHES]
        
    # for
    
    return retval


# get_FBGdata_window
if __name__ == '__main__':
    directory = "../FBG_Needle_Calibaration_Data/needle_1/"
    fbgdata_list = [f.replace( '\\', '/' ) for f in 
                    sorted( glob.glob( directory + "*/fixed_fbgdata_*.txt" ) )]
    needleparam = directory + "needle_params.csv"
    
    print( "Reading in FBG data" )
    ts, fbgdata = read_fbgData( fbgdata_list[0], 3 )
    print( "FBG data read in.\n" )
    print( ts.shape )
    
    print( "Beginning windowed time comparison." )    
    dt = 0.005
    # test 1: using my method
    start = time.time()
    print( np.count_nonzero( get_FBGdata_windows( ts, dt, ts ) + 1 ) )
    elapsed = time.time() - start
    print( f"Elapsed time for my implemenation is {elapsed} seconds." )
    
    # test 2: numpy method
    start = time.time()
    print( np.count_nonzero( get_window_numpy( ts, dt, ts ) + 1 ) )
    elapsed = time.time() - start
    print( f"Elapsed time for numpy implemenation is {elapsed} seconds." )
    
# if
    

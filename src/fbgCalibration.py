'''
Created on Dec 6, 2019

@author: Dimitri Lezcano

@summary: This script is intended for the use of performing the data processing
          incorporating the image and FBG data when needle calibration is 
          performed.
'''

import numpy as np
from datetime import datetime
from _ast import Num


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
                ts = line.split( '\n' )[-1]
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
    
    timestamps = np.empty( 0 )
    fbgdata = np.empty( ( 0, 3 * num_active_areas ) )
    
    with open( filename, 'r' ) as file:
        lines = file.read().split( "\n" )
        
        for line in lines:
            ts, datastring = line.split( ':' )
            timestamps = np.append( timestamps, ts )
            data = np.fromstring( datastring, sep = ',' )
            
            if len( data ) == 3 * num_active_areas:  # all active areas measured
                fbgdata = np.vstack( ( fbgdata, data ) )
                
            else:  # too many or not enough sensor readings
                fbgdata = np.vstack( ( fbgdata, -1 * np.ones( 6 ) ) )
                
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


def get_FBGdata_window( lutime: datetime, dt: float, timestamps: np.ndarray, FBGdata: np.ndarray ):
    """ Function to find the windowed timedata from the gathered FBGdata.
    
        @param lutime: datetime, the lookup time data
        
        @param dt:     float, the time window length
        
        @param timestamps: numpy 1-D array of timestamps
        
        @param FBGdata:    numpy array of the fbg data for each of the timestamps
        
        @return 1-D array -> corresponding timestamps
                2-D array -> FBG data stored in each row.
                
    """
    pass


# get_FBGdata_window
if __name__ == '__main__':
    directory = "../FBG_Needle_Calibaration_Data/needle_1/"
#     filename = "12-09-19_12-29/fbgdata_2019-12-09_12-29-20.txt"
#     filename = "12-09-19_13-34/fbgdata_2019-12-09_13-34-52.txt"
#     filename = "12-09-19_13-49/fbgdata_2019-12-09_13-49-12.txt"
    filename = "12-09-19_14-01/fbgdata_2019-12-09_14-01-06.txt"    
    
    fix_fbgData( directory + filename )
    

#!/usr/local/bin/python3.7

'''
Created on Nov 26, 2019

@author: Dimitri Lezcano

@summary: Script in order to get peaks on loop from the si155 fbg interrogator
          asynchronously using the AsyncHyperion class.
'''

import asyncio
import sys
import time
from datetime import datetime

import numpy as np

from hyperion import AsyncHyperion

TIME_FMT = "%H-%M-%S.%f"
DEFAULT_OUTFILE = "data/%Y-%m-%d_%H-%M-%S.txt"
MAX_CHANNELS = 4


def create_outfile( filename: str = None ):
    if filename:
        retval = filename
        
    else:
        now = datetime.now()
        retval = now.strftime( DEFAULT_OUTFILE )

    # else

    return retval

# crate_outfile


def parsepeakdata( data: dict ):
    """ Method to parse the peak data into a good format for printing """
    timestamp = datetime.fromtimestamp( data["timestamp"] )  # parse timestamp
    peaks = np.zeros( 0 )
    # append the chang
    for channel in range( 1, MAX_CHANNELS + 1 ):
        peaks = np.append( peaks, data['data'][channel] )
    
    # parse timestamp and peak date into str formats 
    str_ts = timestamp.strftime( TIME_FMT )
    
    str_peaks = np.array2string( peaks, precision = 10, separator = ', ',
                                 max_line_width=np.inf )
    str_peaks = str_peaks.strip( '[]' )  # remove the brackets
#    print( str_ts + ": " + str_peaks + '\n' )
    
    return str_ts + ": " + str_peaks + '\n'

# parsepeakdata


async def main( *argv ):
    """ Method to run the script for gathering data from the si155 fbg interrogator. """
    # meant for async data and taken from example file
    loop = asyncio.get_event_loop()
#     queue = asyncio.Queue( maxsize = 5, loop = loop )
    
    # interrogator instantiations
    ipaddress = '10.162.34.16'
    fbginterr = AsyncHyperion( ipaddress, loop )
    await fbginterr.set_ntp_enabled( True )  # change the time protocol

    # output file set up
    global outfile
    arg_1 = None if len( argv ) == 0 else argv[0]
    outfile = create_outfile( arg_1 )
    
    with open( outfile, 'w+' ) as writestream:
        while True:
            peaks = await fbginterr.get_peaks()
            timestamp = datetime.timestamp( datetime.now() )
            data = {"timestamp": timestamp, "data": peaks}
            str_peaks = parsepeakdata( data )
            writestream.write( str_peaks )
#            print( str_peaks )
                
        # while

    # with
        
    return 0
# main

    
if __name__ == '__main__':
    t0 = time.perf_counter()
    try:
        asyncio.run( main( *sys.argv[1:] ) )
    
    except KeyboardInterrupt:
        print("Ending data collection.")
        
    finally:
        print( "\nData collection Terminated." )
        print( f"Peak FBG data file '{outfile}' written." )
        dt = time.perf_counter() - t0
        print( f"Elapsed time for recording data: {dt:.2f}s." )
        
    # finally
        
        
# if

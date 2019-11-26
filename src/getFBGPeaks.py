'''
Created on Nov 26, 2019

@author: Dimitri Lezcano

@summary: Script in order to get peaks on loop from the si155 fbg interrogator
'''

import sys
import numpy as np
import asyncio, timeit, time
from datetime import datetime
from hyperion import Hyperion, HCommTCPPeaksStreamer

TIME_FMT = "%H-%M-%S.%f"
DEFAULT_OUTFILE = "%Y-%m-%d_%H-%M-%S.txt"


def create_outfile( filename: str = None ):
    if filename:
        retval = filename
        
    else:
        now = datetime.now()
        outfile = now.strftime( DEFAULT_OUTFILE )


def parsepeakdata( data: dict ):
    """ Method to parse the peak data into a good format for printing """
    timestamp = datetime.fromtimestamp( data["timestamp"] )  # parse timestamp
    peaks = data["data"][:]
    
    # parse timestamp and peak date into str formats 
    str_ts = timestamp.strftime( TIME_FMT )
    
    str_peaks = np.array2string( peaks, precision = 10, separator = ', ' )
    str_peaks.strip( '[]' )  # remove the brackets
    
    return str_ts + ": " + str_peaks + '\n'

# parsepeakdata


def main( *argv ):
    """ Method to run the script for gathering data from the si155 fbg interrogator. """
    # interrogator instantiations
    ipaddress = ''
#     fbginterr = Hyperion( ipaddress )

    # output file set up
    outfile = create_outfile( argv[0] )
    
    # meant for peak-streaming and taken from example file
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue( maxsize = 5, loop = loop )
    
    # if one would need to check for multiple setups
    serial_numbers = []
    
    # create the streamer object instance
    peaks_streamer = HCommTCPPeaksStreamer( ipaddress, loop, queue )
    t0 = time.perf_counter()
    with open( outfile, 'w+' ) as writestream:

        async def write_peaks():
            while True:
                peak_data = await queue.get()
                queue.task_done()
                
                # check if the streaming is streaming any data
                if peak_data['data']:
                    serial_numbers.append( peak_data['data'].header.serial_number )
                    peak_str = parsepeakdata( peak_data )
                    writestream.write( peak_str )
                    
                # if
                
                else:
                    print( "Writing peak data has ended." )
                    break  # streaming has ended
                
                # else
            
            # while
            
        # async:write_peaks
        
        loop.create_task( write_peaks() )
        try:
            loop.run_until_complete( peaks_streamer.stream_data() )
            
        except KeyboardInterrupt:
            peaks_streamer.stop_streaming()
            print( "Streaming has ended." )
            
    # with
    print( f"Peak FBG data file '{outfile}' written." )
    dt = time.perf_counter - t0
    
    print( f"Elapsed time for recording data: {dt:.2f}s." )
    
# main

    
if __name__ == '__main__':
    main( *sys.argv[1:] )
    
# if

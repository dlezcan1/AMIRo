#!/usr/local/bin/python3.7

'''
Created on Nov 26, 2019

@author: Dimitri Lezcano

@summary: Script in order to get peaks on loop from the si155 fbg interrogator
          using basic calls. Not as reliable as asynchronous one.
'''

import sys
from datetime import datetime

import numpy as np

from hyperion import Hyperion

TIME_FMT = "%H-%M-%S.%f"


def parsepeakdata( data: dict, interrogator: Hyperion ):
    """ Method to parse the peak data into a good format for printing """
    timestamp = datetime.fromtimestamp( data[ "timestamp" ] )  # parse timestamp
    peaks = np.zeros( 0 )
    for channel in range( 1, interrogator.channel_count + 1 ):
        peaks = np.append( peaks, data['data'][channel] )
    
    # parse timestamp and peak date into str formats 
    str_ts = timestamp.strftime( TIME_FMT )
    
    str_peaks = np.array2string( peaks, precision = 10, separator = ', ',
                                 max_line_width=np.inf )
    str_peaks = str_peaks.strip( "[]" )  # remove the brackets
    print( str_ts + ": " + str_peaks + '\n' )
    
    return str_ts + ": " + str_peaks + '\n'

# parsepeakdata


def main( *argv ):
    """ Method to run the script for gathering data from the si155 fbg interrogator. """
    # interrogator instantiations
    ipaddress = '10.162.34.7'
    fbginterr = Hyperion( ipaddress )

    while True:
        try:
            timestamp = datetime.timestamp( datetime.now() )
            data = {"timestamp": timestamp, "data": fbginterr.peaks}
            str_peaks = parsepeakdata( data, fbginterr )
            print( str_peaks )

        # try
        except KeyboardInterrupt:
            print( "Terminating data collection." )
            break

# main


if __name__ == "__main__":
    main( *sys.argv[1:] )

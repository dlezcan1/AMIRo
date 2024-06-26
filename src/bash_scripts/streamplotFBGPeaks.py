'''
Created on Nov 26, 2019

@author: Dimitri Lezcano

@bug: The timing of the script is not 100% accurate with the computer's time
      and there does not appear to be a way to set the time in the script.

@summary: Script in order to get peaks on loop from the si155 fbg interrogator
          asynchronously using the HCommTCPPeaksstreamer.
'''

import sys
import numpy as np
import argparse
import asyncio, timeit, time
from datetime import datetime
from hyperion import Hyperion, HCommTCPPeaksStreamer, HCommTCPSpectrumStreamer
import matplotlib.pyplot as plt

TIME_FMT = "%H-%M-%S.%f"
MAX_CHANNELS = 4

parser = argparse.ArgumentParser( description="Function to get FBG data and write to a log file." )
parser.add_argument( "ip", type=str )


def main( args: argparse.Namespace ):
    """ Method to run the script for gathering data from the si155 fbg interrogator. """
    global MAX_CHANNELS
    # meant for peak-streaming and taken from example file
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue( maxsize=5, loop=loop )

    # if one would need to check for multiple setups
    serial_numbers = [ ]

    # create the streamer object instance
    interrogator = Hyperion( args.ip )
    spectra_streamer = HCommTCPSpectrumStreamer( args.ip, loop, queue )

    # determine channel count and wavelengths being streamed
    MAX_CHANNELS = interrogator.channel_count
    wavelengths = interrogator.spectra.wavelengths

    plt.ion()

    async def stream_spectra():
        global MAX_CHANNELS
	# prepare the plot
        fig = plt.figure()
        ax = fig.add_subplot( 111 )
        lines = { }

        for i in range( 1, MAX_CHANNELS + 1 ):
            lines[ i ], = ax.plot( wavelengths, np.zeros_like( wavelengths ), label=f"CH{i}" )

        # for
        fig.suptile( f"Spectrum of Interrogator: {args.IP}" )
        ax.legend()
        ax.set_xlabel( 'Wavelength (nm)' )
        ax.set_ylabel( 'Power (dB)' )

        while True:
            try:
                spectra_data = await queue.get()
                queue.task_done()
		
                print("Got data.")

                # check if the streaming is streaming any data and the plot window hasn't closed
                if spectra_data[ 'data' ] and plt.fignum_exists( fig.number ):
                    serial_numbers.append( spectra_data[ 'data' ].header.serial_number )

                    # update y-data for all of the plots
                    for ch_i in range( 1, MAX_CHANNELS + 1 ):
                        spectrum_i = spectra_data[ 'data' ][ ch_i ]
                        lines[ ch_i ].set_ydata( spectrum_i )
                        fig.canvas.draw()
                        fig.canvas_flush_events()

                    # for
                # if

                else:
                    spectra_streamer.stop_streaming()
                    break  # streaming has ended

                # else

            except KeyboardInterrupt:
                spectra_streamer.stop_streaming()
                plt.close( 'all' )

            # except
        # while

    # async:write_peaks

    # begin running the spectra streaming
    loop.create_task( stream_spectra() )
    try:
        print( "Running spectrum plotter" )
        loop.run_until_complete( spectra_streamer.stream_data() )
        plt.close( 'all' )

    # try

    except KeyboardInterrupt:
        spectra_streamer.stop_streaming()
        plt.close( 'all' )
        print( "Streaming has ended." )

    # except
# main


if __name__ == '__main__':
    main( parser.parse_args() )

# if

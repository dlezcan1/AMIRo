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
from datetime import datetime
from hyperion import Hyperion
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser( description="Function to get FBG data and write to a log file." )
parser.add_argument( "ip", type=str )


def main( args: argparse.Namespace ):
	""" Method to run the script for gathering data from the si155 fbg interrogator. """
	# create the streamer object instance
	interrogator = Hyperion( args.ip )

	# determine channel count and wavelengths being streamed
	max_channels = interrogator.channel_count
	wavelengths = interrogator.spectra.wavelengths

	plt.ion()

	fig = plt.figure()
	ax = fig.add_subplot( 111 )
	lines = { }

	for i in range( 1, max_channels + 1 ):
		lines[ i ], = ax.plot( wavelengths, np.zeros_like( wavelengths ), label=f"CH{i}" )

		# for
		fig.suptitle( f"Spectrum of Interrogator: {args.ip}" )
		ax.legend()
		ax.set_xlabel( 'Wavelength (nm)' )
		ax.set_ylabel( 'Power (dB)' )
		ax.set_ylim([-80,0])

	while True:
		spectra_data = interrogator.spectra

		if not plt.fignum_exists(fig.number):
			break
		
		# update y-data for all of the plots
		for ch_i in range( 1, max_channels + 1 ):
			spectrum_i = spectra_data[ ch_i ]
			lines[ ch_i ].set_ydata( spectrum_i )
		# for
		fig.canvas.draw()
		fig.canvas.flush_events()

	# while
# main


if __name__ == '__main__':
	main( parser.parse_args() )

# if

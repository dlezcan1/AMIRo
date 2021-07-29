"""
Created on Jul 28, 2021

@author: Dimitri Lezcano

@summary: This is a library to properly open the files for this project.

"""

import numpy as np
import pandas as pd

from FBGNeedle import FBGNeedle


def read_fbgdata( filename: str, num_channels: int, num_active_areas: int ) -> pd.DataFrame:
    """ Read in the FBG data file
        Args:
            filename: str of the filename to be loaded
            num_channels: int of the number of channels to expect
            num_active_areas: int of the number of active areas to expect
    """
    # read in the FBG data file
    with open( filename, 'r' ) as file:
        lines = file.readlines()

    # with

    # generate column names
    ch_aa, channels, active_areas = FBGNeedle.generate_ch_aa( num_channels, num_active_areas )
    columns = [ 'time' ] + ch_aa

    # generate empty data table to return
    ret_val = pd.DataFrame( np.zeros( (0, 1 + num_channels * num_active_areas) ),
                            columns=columns
                            )
    ret_val = ret_val.astype( { 'time': 'datetime64[ns]' } )

    # iterate through each of the lines
    for line in lines:
        ts, signals = line.split( ':' )
        ts = pd.to_datetime( ts, format="%H-%M-%S.%f" )
        signals = np.fromstring( signals, sep=',' )

        # make sure we have the correct number of signals
        if signals.size != num_channels * num_active_areas:
            continue
        # if
        new_row = pd.DataFrame( np.hstack( (ts, signals) ).reshape( 1, -1 ), columns=columns )
        ret_val = ret_val.append( new_row, ignore_index=True )

    # for

    return ret_val


# read_fbgdata

if __name__ == "__main__":
    pass

# if __main__

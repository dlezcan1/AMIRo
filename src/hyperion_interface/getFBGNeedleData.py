import argparse
import numpy as np
import asyncio
import os
import time

from datetime import datetime
from hyperion import Hyperion, HCommTCPPeaksStreamer

OUTFILE_FMT = "fbgdata_%Y-%m-%d_%H-%M-%S.csv"
OUTDIR_FMT = "%m-%d-%y_%H-%M"
TIME_FMT = "%H-%M-%S.%f"
NUM_CHANNELS = 4


def __get_parser():
    parser = argparse.ArgumentParser( description="Gather Data for FBG needle calibration and validation." )

    parser.add_argument( 'K', type=float, help='Curvature of the groove.' )
    parser.add_argument( 'ip', type=str, help="IP address of Hyperion interrogator" )
    parser.add_argument( '-v', '--verbose', action='store_true' )
    parser.add_argument( '-N', '--number-peaks', type=int, default=200, dest='N',
                         help='Number of FBG peaks to gather per dataset' )
    parser.add_argument( '-d', '--directory', type=str, default='', dest='dir', help='Main data directory to save in' )

    return parser


# __get_parser

def create_fbgfilename( filename: str ):
    return datetime.now().strftime( filename )


# create_fbgfilename

def parse_peak_data( data: dict ):
    # get the peak data
    ts = datetime.fromtimestamp( data[ 'timestamp' ] )
    peaks = np.append( [ ], [ data[ 'data' ][ ch_i ] for ch_i in range( 1, NUM_CHANNELS + 1 ) ] )

    # convert to a string line
    str_ts = ts.strftime( TIME_FMT )
    str_peaks = np.array2string( peaks, precision=10, separator=', ', max_line_width=np.inf ).strip( '[]' )

    return str_ts + ": " + str_peaks


# parse_peak_data

async def begin_trial( task, loop ):
    trial = loop.create_task( task )
    await trial


# begin_trial

async def main( args=None ):
    pargs = __get_parser().parse_args( args )

    # set-up interrogator
    print( "Connecting to interrogator at {}...".format( pargs.ip ), end=' ' )
    interrogator = Hyperion( pargs.ip )
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue( maxsize=5 )
    peak_streamer = HCommTCPPeaksStreamer( pargs.ip, loop, queue )
    print( " Connected." )

    # set-up data directory
    out_dir = os.path.join( os.path.normpath( pargs.dir ), str( pargs.K ) )
    out_dir = os.path.join( out_dir, datetime.now().strftime( OUTDIR_FMT ) )
    if not os.path.isdir( out_dir ):
        os.makedirs( out_dir )

    # if
    print( "Data directory: {}".format( out_dir ) )
    print()

    # define write_peak function
    async def write_peaks( outfile ):
        count = 0
        with open( outfile, 'w' ) as writestream:
            while True:
                try:
                    # get the peak data
                    peak_data = await asyncio.wait_for( queue.get(), timeout=5.0 )
                    queue.task_done()

                    if peak_data[ 'data' ]:
                        peak_line = parse_peak_data( peak_data )
                        count += 1
                        if pargs.verbose:
                            print( peak_line )

                        # if
                        writestream.write( peak_line + "\n" )

                    # if
                    else:
                        print( "Writing peak data has ended" )
                        peak_streamer.stop_streaming()
                        break

                    # else

                    if count > pargs.N:  # saved file
                        print( f"Wrote FBG data file: {outfile}" )
                        break

                    # if
                # try
                except KeyboardInterrupt:
                    peak_streamer.stop_streaming()
                    break

                # except KeyboardInterrupt

                except asyncio.exceptions.TimeoutError:
                    peak_streamer.stop_streaming()
                    print("FBG Peak Timeout error")
                    break

                # except TimeoutError

            # while
        # with



    # write_peaks

    # iterate over trials to get number of peaks
    print( "Gathering peak values" )
    peak_task = loop.create_task( peak_streamer.stream_data() )  # run peak streamer
    counter = 1
    while True:
        try:
            if not peak_task.done():
                t0 = time.perf_counter()
                out_file = create_fbgfilename( OUTFILE_FMT )
                out_file = os.path.join( out_dir, out_file )
                trial = loop.create_task( write_peaks( out_file ) )
                print( "Trial {}".format( counter ) )
                await trial
                dt = time.perf_counter() - t0
                print( "Trial {} complete".format( counter ) )
                print( f"Elapsed time for recording data: {dt:.3f}s" )
                input( 'Press [ENTER] to continue or [CTRL-C] + [ENTER] to quit.' )
                counter += 1
                print()

            else:
                break

        # try
        except KeyboardInterrupt:
            peak_streamer.stop_streaming()
            break

        # except
    # while

    print( "Program terminated." )


# main

if __name__ == "__main__":
    asyncio.run( main() )

# if __main__

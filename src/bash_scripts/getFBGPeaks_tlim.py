#!/usr/local/bin/python3.7

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
from datetime import datetime, timedelta
from hyperion import HCommTCPPeaksStreamer

TIME_FMT = "%H-%M-%S.%f"
DEFAULT_OUTFILE = "fbgdata_%Y-%m-%d_%H-%M-%S.txt"
MAX_CHANNELS = 4

parser = argparse.ArgumentParser(description="Function to get FBG data and write to a log file.")
parser.add_argument("ip", nargs="?", type=str, default="10.162.34.16")
parser.add_argument("-o", "--output", type=str, dest="outfile", default=None)
parser.add_argument('-v', '--verbose', action="store_true")
parser.add_argument('-d', '--directory', type=str, default='', dest='dir')


def create_outfile(filename: str = None):
    if filename:
        retval = filename

    else:
        now = datetime.now()
        retval = now.strftime(DEFAULT_OUTFILE)

    # else

    return retval


# create_outfile

def parsepeakdata(data: dict):
    """ Method to parse the peak data into a good format for printing """
    timestamp = datetime.fromtimestamp(data["timestamp"])  # parse timestamp
    peaks = np.zeros(0)
    # append the chang
    for channel in range(1, MAX_CHANNELS + 1):
        peaks = np.append(peaks, data['data'][channel])

    # parse timestamp and peak date into str formats
    str_ts = timestamp.strftime(TIME_FMT)

    str_peaks = np.array2string(peaks, precision=10, separator=', ')
    str_peaks = str_peaks.strip('[]')  # remove the brackets

    return str_ts + ": " + str_peaks + '\n'


# parsepeakdata

def main2( args: argparse.Namespace):
    dt = timedelta(seconds = 30)
    ipaddress = args.ip

    global DEFAULT_OUTFILE
    DEFAULT_OUTFILE = args.dir + DEFAULT_OUTFILE
    dt_file = timedelta(minutes=1)

    outfile = create_outfile(args.outfile)
    print(f"Opening a file to write FBG data to: {outfile}")

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=5, loop=loop)

    # if one would need to check for multiple setups
    serial_numbers = []

    # create the streamer object instance
    peaks_streamer = HCommTCPPeaksStreamer(ipaddress, loop, queue)
    t0 = time.perf_counter()

    writefile = open(outfile, 'w+')

    async def write_peaks():
        nonlocal writefile
        t_begin = datetime.now()
        while True:
            try:
                peak_data = await queue.get()
                queue.task_done()

                # check if the streaming is streaming any data
                if peak_data['data']:
                    serial_numbers.append(peak_data['data'].header.serial_number)
                    peak_str = parsepeakdata(peak_data)
                    if args.verbose:
                        print(peak_str)

                    writefile.write(peak_str)
                # if

                else:
                    print("Writing peak data has ended.")
                    break  # streaming has ended

                # else

                if datetime.now() - t_begin > dt:
                    writefile.close()
                    outfile = create_outfile(args.outfile)
                    writefile = open(outfile, 'w+')
                    if args.verbose:
                        print(f"Opening a new file to write FBG data to: {writefile}")

                    t_begin = datetime.now()

                # if

            except KeyboardInterrupt:
                peaks_streamer.stop_streaming()

            # except
        # while
    # async:write_peaks

    loop.create_task(write_peaks())
    try:
        print("Gathering peak values")
        loop.run_until_complete(peaks_streamer.stream_data())

    except KeyboardInterrupt:
        peaks_streamer.stop_streaming()
        print("Streaming has ended.")

    # with
    print(f"Peak FBG data file '{outfile}' written.")
    dt = time.perf_counter() - t0

    print(f"Elapsed time for recording data: {dt:.2f}s.")

# main2


def main(args: argparse.Namespace):
    """ Method to run the script for gathering data from the si155 fbg interrogator. """
    # interrogator instantiations
    ipaddress = args.ip

    global DEFAULT_OUTFILE

    DEFAULT_OUTFILE = args.dir + DEFAULT_OUTFILE

    # output file set up
    #    arg_1 = None if len( argv ) == 0 else argv[0]
    outfile = create_outfile(args.outfile)

    # meant for peak-streaming and taken from example file
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue(maxsize=5, loop=loop)

    # if one would need to check for multiple setups
    serial_numbers = []

    # create the streamer object instance
    peaks_streamer = HCommTCPPeaksStreamer(ipaddress, loop, queue)
    t0 = time.perf_counter()

    with open(outfile, 'w+') as writestream:

        async def write_peaks():
            while True:
                try:
                    peak_data = await queue.get()
                    queue.task_done()

                    # check if the streaming is streaming any data
                    if peak_data['data']:
                        serial_numbers.append(peak_data['data'].header.serial_number)
                        peak_str = parsepeakdata(peak_data)
                        if args.verbose:
                            print(peak_str)

                        writestream.write(peak_str)
                    # if

                    else:
                        print("Writing peak data has ended.")
                        break  # streaming has ended

                    # else

                except KeyboardInterrupt:
                    peaks_streamer.stop_streaming()

                # except

            # while

        # async:write_peaks

        loop.create_task(write_peaks())
        try:
            print("Gathering peak values")
            loop.run_until_complete(peaks_streamer.stream_data())

        except KeyboardInterrupt:
            peaks_streamer.stop_streaming()
            print("Streaming has ended.")

    # with
    print(f"Peak FBG data file '{outfile}' written.")
    dt = time.perf_counter() - t0

    print(f"Elapsed time for recording data: {dt:.2f}s.")


# main


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.outfile:
        main2(args)

    else:
        main(args)

# if

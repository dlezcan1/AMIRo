import sys
import numpy as np
import asyncio, timeit, time
from datetime import datetime
from hyperion import Hyperion, HCommTCPPeaksStreamer

TIME_FMT = "%H-%M-%S.%f"    

def parsepeakdata( data: dict , interrogator: Hyperion):
    """ Method to parse the peak data into a good format for printing """
    timestamp = datetime.fromtimestamp( data["timestamp"] )  # parse timestamp
    peaks = np.zeros(0)
    for channel in range(1,interrogator.channel_count + 1):
        peaks = np.append(peaks, data['data'][channel])
    
    # parse timestamp and peak date into str formats 
    str_ts = timestamp.strftime( TIME_FMT )
    
    str_peaks = np.array2string( peaks, precision = 10, separator = ', ' )
    str_peaks = str_peaks.strip("[]")  # remove the brackets
    print(str_ts + ": " + str_peaks + '\n')
    
    return str_ts + ": " + str_peaks + '\n'

# parsepeakdata


def main():
    # interrogator instantiations
    ipaddress = '10.162.34.7'
    fbginterr = Hyperion( ipaddress )

    # meant for peak-streaming and taken from example file
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue( maxsize = 5, loop = loop )
    
    # if one would need to check for multiple setups
    serial_numbers = []
    
    # create the streamer object instance
    peaks_streamer = HCommTCPPeaksStreamer( ipaddress, loop, queue )
    t0 = time.perf_counter()

    
    while True:
        try:
            data = {"timestamp": 0, "data": fbginterr.peaks}
            str_peaks = parsepeakdata(data, fbginterr)
            print(str_peaks)

        # try
        except KeyboardInterrupt:
            print("Terminating data collection.")
            break

# main

if __name__ == "__main__":
    main()

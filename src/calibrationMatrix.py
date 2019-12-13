'''
Created on Dec 12, 2019

@summary: This script is intended to generate the calibration matrices relating
          curvature to FBG sensor readings
'''

import numpy as np
import matplotlib.pyplot as plt
import glob, re
import time  # for debugging purposes

def load_curvature(directory):
    '''loads all the curvature_monofbg text files
        combines curvature results into one n x 4 numpy array
        with the first column with the timestamp data

        Output: nx4 numpy array of floats
    '''
    curvature = np.empty([0,4])

    name_length = len(directory + "curvature_mono_mm_dd_yy_")
    filenames = glob.glob(directory + "curvature_mono_12*.txt")
    print('number of files: %s' %len(filenames))

    for file in filenames:
        with open(file,'r') as f:
            for i, line in enumerate(f):
                if i == 3:
                    timestamp = file[name_length:-4]
                    hour, minute, sec = timestamp.split('-')
                    timeInSec = float(hour)*3600 + float(minute)*60 + float(sec)

                    data = [number.strip().split(',') for number in line.strip().split(":")]
                    toappend = [float(i) for i in data[1]] #convert to floats
                    toappend.insert(0, timeInSec)
                    
                    curvature = np.vstack([curvature, toappend])
    print(curvature.shape)
    return curvature


def sync_fbg(directory, curvature, w1, w2):
    '''loads fbgdata text file
        calculates baseline FBG readings using the first 100 lines
        finds closest line that matches with each curvature file
        and takes average wavelength based on window size (2*w+1 points)
    '''
    global startTime
    name_length = len(directory + "fixed_fbgdata_yyyy_mm_dd_")
    filenames = glob.glob(directory + "fixed_fbgdata_*.txt")
    curv_idx = 0

    ## generate a numpy array of rawFBG data
    rawFBG = np.empty([0,10])
    if len(filenames) == 1:
        for file in filenames:
            baseTime = file[name_length:-4]
            hour, minute, sec = baseTime.split('-')
            baseInSec = float(hour)*3600 + float(minute)*60 + float(sec)

            with open(file,'r') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        data = [number.strip().split(',') for number in line.strip().split(":")]
                        hour, minute, sec = data[0][0].split('-')
                        timeInSec = float(hour)*3600 + float(minute)*60 + float(sec)
                        offset = baseInSec - timeInSec
                        print('offset: %s' %offset)

                    data = [number.strip().split(',') for number in line.strip().split(":")]
                    hour, minute, sec = data[0][0].split('-')
                    timeInSec = float(hour)*3600 + float(minute)*60 + float(sec) + offset

                    if abs(timeInSec - curvature[curv_idx,0]) < w1:
                        # print('appending for %s' %curv_idx)
                        toappend = [float(i) for i in data[1]] #convert to floats
                        toappend.insert(0, timeInSec)

                        rawFBG = np.vstack([rawFBG, toappend])
                    
                    if (timeInSec - curvature[curv_idx,0]) >= w1:
                        if curv_idx < curvature.shape[0]-1:
                            curv_idx += 1

    ## generate baseline
    numLines = 200
    baseline = np.sum(rawFBG[0:numLines, 1:], axis=0)/float(numLines)

    ## sync FBG with curvature timestamps
    avgFBG = np.empty([0,10])
    for time in curvature[:,0]:
        match_idx = np.argmin(abs(rawFBG[:,0]-time))
        avg = np.sum(rawFBG[match_idx-w2:match_idx+w2, 1:], axis=0)/(2*w2+1)
        
        toappend = np.hstack([rawFBG[match_idx,0], avg])
        avgFBG = np.vstack([avgFBG, toappend])

    print("difference in number of curvatures to average FBG: ")
    print(curvature.shape[0]-avgFBG.shape[0])

    return baseline, avgFBG


def wavelength_shift(avg_fbg, baseline):
    '''process averaged FBG wavelength readings
    '''

def leastsq_fit(delta_fbg, curvature):
    ''' computes least squares fit between curvature and fbg data
    '''

def plot(delta_fbg, curvature):
    '''plots delta_fbg vs. curvature, just to see
    '''

def main():
    root_path = "../FBG_Needle_Calibration_Data/needle_1/"
    # root_path = "C:/Users/epyan/Documents/JHU/Research/Shape Sensing/FBG_Needle_Calibration_Data/needle_1/"
    folder = "12-09-19_12-29/"
    directory = root_path + folder
    w1 = 1 # number of +/- seconds of data to keep
    w2 = 20 # window size for averaging FBG data

    curvature = load_curvature(directory)
    # print(curvature)
    startTime = time.time()
    baseline, avgFBG = sync_fbg(directory, curvature, w1, w2)
    # print(baseline)
    # print(avgFBG)
    np.savetxt(directory + 'filteredFBG.txt', avgFBG)
    print('time to sync: %s' %(time.time()-startTime))

if __name__ == '__main__':
    main()


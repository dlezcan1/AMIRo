'''
Created on Dec 12, 2019

@summary: This script is intended to generate the calibration matrices relating
          curvature to FBG sensor readings
'''

import numpy as np
import matplotlib.pyplot as plt
import glob, re
import time  # for debugging purposes
from datetime import timedelta, datetime
from _pickle import load


def load_curvature( directory ):
    '''loads all the curvature_monofbg text files
        combines curvature results into one n x 4 numpy array
        with the first column with the timestamp data

        Output: nx4 numpy array of floats
    '''
    curvature = np.empty( [0, 4] )

    name_length = len( directory + "curvature_monofbg_mm_dd_yyyy_" )
    filenames = glob.glob( directory + "curvature_monofbg*.txt" )
    print( 'number of files: %s' % len( filenames ) )

    for file in filenames:
        with open( file, 'r' ) as f:
            for i, line in enumerate( f ):
                if i == 3:
                    timestamp = file[name_length:-4]
                    hour, minute, sec = timestamp.split( '-' )
                    timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )

                    data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                    toappend = [float( i ) for i in data[1]]  # convert to floats
                    toappend.insert( 0, timeInSec )
                    
                    curvature = np.vstack( [curvature, toappend] )
    print( curvature.shape )
    return curvature

# load_curvature


def sync_fbg( directory, curvature, w1, w2 ):
    '''loads fbgdata text file
        calculates baseline FBG readings using the first 100 lines
        finds closest line that matches with each curvature file
        and takes average wavelength based on window size (2*w+1 points)
    '''
#     global startTime
    name_length = len( directory + "fixed_fbgdata_yyyy_mm_dd_" )
    filenames = glob.glob( directory + "fixed_fbgdata_*.txt" )
    curv_idx = 0
    camera_time_offset = 0.75  # seconds

    # # generate a numpy array of rawFBG data
    rawFBG = np.empty( [0, 10] )
    if len( filenames ) == 1:
        for file in filenames:
            baseTime = file[name_length:-4]
            hour, minute, sec = baseTime.split( '-' )
            baseInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )

            with open( file, 'r' ) as f:
                for i, line in enumerate( f ):
                    if i == 0:
                        data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                        hour, minute, sec = data[0][0].split( '-' )
                        timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )
                        offset = baseInSec - timeInSec
                        print( 'offset: %s' % offset )

                    data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                    hour, minute, sec = data[0][0].split( '-' )
                    timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec ) + offset

                    if abs( timeInSec - curvature[curv_idx, 0] - camera_time_offset ) < w1:
                        # print('appending for %s' %curv_idx)
                        toappend = [float( i ) for i in data[1]]  # convert to floats
                        toappend.insert( 0, timeInSec )

                        rawFBG = np.vstack( [rawFBG, toappend] )
                    
                    if ( timeInSec - curvature[curv_idx, 0] ) >= w1:
                        if curv_idx < curvature.shape[0] - 1:
                            curv_idx += 1

    # # generate baseline
    numLines = 200
    baseline = np.sum( rawFBG[0:numLines, 1:], axis = 0 ) / float( numLines )

    # # sync FBG with curvature timestamps
    avgFBG = np.empty( [0, 10] )
    for time in curvature[:, 0]:
        match_idx = np.argmin( abs( rawFBG[:, 0] - time ) )
        avg = np.sum( rawFBG[match_idx - w2:match_idx + w2, 1:], axis = 0 ) / ( 2 * w2 + 1 )
        
        toappend = np.hstack( [rawFBG[match_idx, 0], avg] )
        avgFBG = np.vstack( [avgFBG, toappend] )

    print( "difference in number of curvatures to average FBG: " )
    print( curvature.shape[0] - avgFBG.shape[0] )

    return baseline, avgFBG

# sync_fbg

def process_fbg(directory):
    ''' Loads all FBG files, average_fbg and baseline
        rawFBG_list is a list of numpy arrays with each array holding the raw data from one file
    '''
    name_length = len( directory + "fbgdata_yyyy_mm_dd_" )
    filenames = glob.glob( directory + "fbgdata*.txt" )
    print( 'number of files: %s' % len( filenames ) )

    rawFBG = np.empty( [0, 10] )
    avgFBG = np.empty([0,10])
    rawFBG_list = []

    for idx, file in enumerate(filenames):
        baseTime = file[name_length:-4]
        hour, minute, sec = baseTime.split( '-' )
        baseInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )

        with open(file, 'r') as f:
            for i, line in enumerate( f ):
                if i == 0:
                    data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                    hour, minute, sec = data[0][0].split( '-' )
                    timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )
                    offset = baseInSec - np.floor(timeInSec)
                    print( 'offset: %s' % offset )

                data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                hour, minute, sec = data[0][0].split( '-' )
                timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec ) + offset

                toappend = [float( i ) for i in data[1]]  # convert to floats
                toappend.insert( 0, timeInSec )
                rawFBG = np.vstack( [rawFBG, toappend] )

        ## create baseline using first file
        if idx == 0:
            baseline = np.mean(rawFBG[:, 1:], axis=0)

        rawFBG_list.append(rawFBG)
        avg = np.mean(rawFBG[:, 1:], axis=0)
        toappend = np.hstack([baseInSec, avg])
        avgFBG = np.vstack([avgFBG, toappend])
        rawFBG = np.empty([0, 10])
    
    return rawFBG_list, avgFBG, baseline

# process_fbg


def wavelength_shift( avg_fbg, baseline ):
    '''process averaged FBG wavelength readings
    '''
    aa1_idxs = [0, 3, 6]
    aa2_idxs = [1, 4, 7]
    aa3_idxs = [2, 5, 8]
    
#     wl_aa1 = avg_fbg[:, aa1_idxs]
#     wl_aa2 = avg_fbg[:, aa2_idxs]
#     wl_aa3 = avg_fbg[:, aa3_idxs]

    deltaFBG = avg_fbg[:, 1:] - baseline # wavelength shift

    # average shift at each active area
    mean_aa1 = np.mean(deltaFBG[:, aa1_idxs], axis = 1)
    mean_aa2 = np.mean(deltaFBG[:, aa2_idxs], axis = 1)
    mean_aa3 = np.mean(deltaFBG[:, aa3_idxs], axis = 1)

    # subtract contribution from temperature (uniform across each active area)
    deltaFBG[:, aa1_idxs] -= mean_aa1.reshape( -1, 1 )
    deltaFBG[:, aa2_idxs] -= mean_aa2.reshape( -1, 1 )
    deltaFBG[:, aa3_idxs] -= mean_aa3.reshape( -1, 1 )
    
    # baselines_aa1 = np.mean( delta_fbg[:, aa1_idxs], axis = 1 )
    # baselines_aa2 = np.mean( delta_fbg[:, aa2_idxs], axis = 1 )
    # baselines_aa3 = np.mean( delta_fbg[:, aa3_idxs], axis = 1 )
    
    # delta_fbg[:, aa1_idxs] = delta_fbg[:, aa1_idxs] - baselines_aa1.reshape( -1, 1 )
    # delta_fbg[:, aa2_idxs] = delta_fbg[:, aa2_idxs] - baselines_aa2.reshape( -1, 1 )
    # delta_fbg[:, aa3_idxs] = delta_fbg[:, aa3_idxs] - baselines_aa3.reshape( -1, 1 )
    
    # baselines_aa1 = np.mean( avg_fbg[:, aa1_idxs], axis = 1 )
    # baselines_aa2 = np.mean( avg_fbg[:, aa2_idxs], axis = 1 )
    # baselines_aa3 = np.mean( avg_fbg[:, aa3_idxs], axis = 1 )
    
    # avg_fbg[:, aa1_idxs] -= baselines_aa1.reshape( -1, 1 )
    # avg_fbg[:, aa2_idxs] -= baselines_aa2.reshape( -1, 1 )
    # avg_fbg[:, aa3_idxs] -= baselines_aa3.reshape( -1, 1 )
    
    return deltaFBG

# wavelength_shift


def get_curvature_vectors( curvature, direction ):
    """ Computes the curvature vectors for each of the active areas 
    
        @param direction: the direction vector of the deformation
    """
    ts = np.empty( 0 )
    aa1 = np.empty( ( 0, 3 ) )
    aa2 = np.empty( ( 0, 3 ) )
    aa3 = np.empty( ( 0, 3 ) )
    
    for t, k1, k2, k3 in curvature:
        ts = np.append( ts, t )
        aa1 = np.vstack( ( aa1, k1 * direction ) )
        aa2 = np.vstack( ( aa2, k2 * direction ) )
        aa3 = np.vstack( ( aa3, k3 * direction ) )
        
    # for
    
    return [ts, aa1, aa2, aa3]

# get_curvature_vectors


def load_filteredfbg_data( filename: str ):
    """ Method to read in the fbg filtered data """
    inp = np.loadtxt( filename, np.float64 )
    
    ts = inp[:, 0]
    data = inp[:, 1:]
    
    return ts, data

# load_filteredfbg_data


def leastsq_fit( delta_fbg, curvature ):
    ''' computes least squares fit between curvature and fbg data
    
        @param curvature: lists of numpy arrays of the vectors of curvatures,
                          row-wise
                          (i.e. [ array([ kj,1 ]), array([ kj,2 ]),...)
                          
        Want to find C_i such that
        
          ...       ...      ...                ...     ...     ...
        ( dl1,i    dl2,i    dl3,i ) C_i^T   = ( k1,i    k2,i    k3,i )
          ...       ...      ...                ...     ...     ...

        where C_i is for the i-th active area.
    '''
    # seperate each of the active areas
    aa1_idxs = [0, 3, 6]
    aa2_idxs = [1, 4, 7]
    aa3_idxs = [2, 5, 8]
    delta_aa1 = delta_fbg[:, aa1_idxs]
    delta_aa2 = delta_fbg[:, aa2_idxs]
    delta_aa3 = delta_fbg[:, aa3_idxs]
    
    curvature1 = curvature[0]
    curvature2 = curvature[1]
    curvature3 = curvature[2]
    
    C1, resid1, rnk1, sng1 = np.linalg.lstsq( delta_aa1, curvature1, None )
    C2, resid2, rnk2, sng2 = np.linalg.lstsq( delta_aa2, curvature2, None )
    C3, resid3, rnk3, sng3 = np.linalg.lstsq( delta_aa3, curvature3, None )
    
    delta1 = curvature1 - np.dot( delta_aa1, C1 )
    delta2 = curvature2 - np.dot( delta_aa2, C2 )
    delta3 = curvature3 - np.dot( delta_aa3, C3 )
    resid1 = np.linalg.norm( delta1, axis = 0 )
    resid2 = np.linalg.norm( delta2, axis = 0 )
    resid3 = np.linalg.norm( delta3, axis = 0 )
    
#     rel_err1 = np.divide(delta1 , curvature1, out=np.zeros_like(delta1), where=curvature1!=0)
    rel_err1 = delta1 / curvature1
    rel_err2 = delta2 / curvature2
    rel_err3 = delta3 / curvature3
    
    rel_err1[np.logical_or( rel_err1 == -np.inf, rel_err1 == np.inf )] = np.nan
    rel_err2[np.logical_or( rel_err2 == -np.inf, rel_err2 == np.inf )] = np.nan
    rel_err3[np.logical_or( rel_err3 == -np.inf, rel_err3 == np.inf )] = np.nan
    
    min_relerr1 = np.nanmin( rel_err1 , axis = 0 )
    mean_relerr1 = np.nanmean( rel_err1, axis = 0 )
    max_relerr1 = np.nanmax( rel_err1 , axis = 0 )
    
    min_relerr2 = np.nanmin( rel_err2, axis = 0 )
    mean_relerr2 = np.nanmean( rel_err2, axis = 0 )
    max_relerr2 = np.nanmax( rel_err2, axis = 0 )
    
    min_relerr3 = np.nanmin( rel_err3 , axis = 0 )
    mean_relerr3 = np.nanmean( rel_err3 , axis = 0 )
    max_relerr3 = np.nanmax( rel_err3 , axis = 0 )
    
    print( f"1) Residuals: {resid1}" )
    print( f"Relative error\nMin: {min_relerr1}\nMean: {mean_relerr1}\nMax: {max_relerr1}\n\n" )
    
    print( f"2) Residuals: {resid2}" )
    print( f"Relative error\nMin: {min_relerr2}\nMean: {mean_relerr2}\nMax: {max_relerr2}\n\n" )
    
    print( f"3) Residuals: {resid3}" )
    print( f"Relative error\nMin: {min_relerr3}\nMean: {mean_relerr3}\nMax: {max_relerr3}\n\n" )
    
    return [np.transpose( C1 ), np.transpose( C2 ), np.transpose( C3 )]
    
# leastsq_fit


def plot( delta_fbg, curvature ):
    '''plots delta_fbg vs. curvature, just to see
    '''
    
# plot


def write_calibration_matrices( outfile, C1, C2, C3 ):
    """ Method to simplify writing the calibration matrices to a file"""
    with open( outfile, 'a' ) as writestream:
        # format the matrices
        msg1 = np.array2string( C1, separator = ',' ).replace( '[', '' ).replace( ']', '' )
        msg2 = np.array2string( C2, separator = ',' ).replace( '[', '' ).replace( ']', '' )
        msg3 = np.array2string( C3, separator = ',' ).replace( '[', '' ).replace( ']', '' )
        
        # write the data
        writestream.write( "C1:\n" + msg1 + '\n\n' )
        writestream.write( "C2:\n" + msg2 + '\n\n' )
        writestream.write( "C3:\n" + msg3 + '\n\n' )
        
    # with
    
    return 0

# write_calibration_matrices
        

def main():
    e1 = np.array( [1, 0, 0] )
    e2 = np.array( [0, 1, 0] )
    root_path = "../FBG_Needle_Calibration_Data/needle_1/"
    # root_path = "C:/Users/epyan/Documents/JHU/Research/Shape Sensing/FBG_Needle_Calibration_Data/needle_1/"
    folder_py = root_path + "12-09-19_12-29/"  # positive y-axis
    folder_px = root_path + "12-09-19_13-34/"  # positive x-axis
    folder_my = root_path + "12-09-19_13-49/"  # negative y-axis
    folder_mx = root_path + "12-09-19_14-01/"  # negative x-axis
    w1 = 1  # number of +/- seconds of data to keep
    w2 = 20  # window size for averaging FBG data

    # load the curvature data
    curvature_py = load_curvature( folder_py )
    curvature_px = load_curvature( folder_px )
    curvature_my = load_curvature( folder_my )
    curvature_mx = load_curvature( folder_mx )
#     print(curvature)

    # load the fbg data
#     startTime = time.time()

    if True:
        baseline_py, avgFBG_py = sync_fbg( folder_py, curvature_py, w1, w2 )
        baseline_px, avgFBG_px = sync_fbg( folder_px, curvature_px, w1, w2 )
        baseline_my, avgFBG_my = sync_fbg( folder_my, curvature_my, w1, w2 )
        baseline_mx, avgFBG_mx = sync_fbg( folder_mx, curvature_mx, w1, w2 )
        
        # save the averaged FBG data
        np.savetxt( folder_py + 'filteredFBG.txt', avgFBG_py )
        np.savetxt( folder_px + 'filteredFBG.txt', avgFBG_px )
        np.savetxt( folder_my + 'filteredFBG.txt', avgFBG_my )
        np.savetxt( folder_mx + 'filteredFBG.txt', avgFBG_mx )
        
        avgFBG_py = avgFBG_py[:, 1:]
        avgFBG_px = avgFBG_px[:, 1:]
        avgFBG_my = avgFBG_my[:, 1:]
        avgFBG_mx = avgFBG_mx[:, 1:]
    
    # if
    
    else:
        _, avgFBG_py = load_filteredfbg_data( folder_py + "filteredFBG.txt" )
        _, avgFBG_px = load_filteredfbg_data( folder_px + "filteredFBG.txt" )
        _, avgFBG_my = load_filteredfbg_data( folder_my + "filteredFBG.txt" )
        _, avgFBG_mx = load_filteredfbg_data( folder_mx + "filteredFBG.txt" )
        print( avgFBG_py.shape )
        print( avgFBG_py )
    
    # else
    
    # compute the delta fbg data
    deltafbg_py = wavelength_shift( avgFBG_py, 0 )
    deltafbg_px = wavelength_shift( avgFBG_px, 0 )
    deltafbg_my = wavelength_shift( avgFBG_my, 0 )
    deltafbg_mx = wavelength_shift( avgFBG_mx, 0 )
    
#     print(baseline)
#     print(avgFBG)
    
    # convert the curvatures to vectors
    curv_vect_py = get_curvature_vectors( curvature_py, e2 )
    curv_vect_px = get_curvature_vectors( curvature_px, e1 )
    curv_vect_my = get_curvature_vectors( curvature_my, -e2 )
    curv_vect_mx = get_curvature_vectors( curvature_mx, -e1 )
    
    # concatenate the data
    
    curv_vect_aa1 = np.vstack( ( curv_vect_py[1], curv_vect_px[1],
                                 curv_vect_my[1], curv_vect_mx[1] ) )
    curv_vect_aa2 = np.vstack( ( curv_vect_py[2], curv_vect_px[2],
                                 curv_vect_my[2], curv_vect_mx[2] ) )
    curv_vect_aa3 = np.vstack( ( curv_vect_py[3], curv_vect_px[3],
                                 curv_vect_my[3], curv_vect_mx[3] ) )
    
    curv_vect = [curv_vect_aa1, curv_vect_aa2, curv_vect_aa3]
    delta_fbg = np.vstack( ( deltafbg_py, deltafbg_px, deltafbg_my, deltafbg_mx ) )
    
    write_vect = np.hstack( ( curv_vect_aa1, curv_vect_aa2, curv_vect_aa3 ) )
    np.savetxt( root_path + "curvature_vectors.txt", write_vect )
    np.savetxt( root_path + "wavelength_shift.txt", delta_fbg )
    
    C1, C2, C3 = leastsq_fit( delta_fbg, curv_vect )
#     return;
    outfile = root_path + "needle_params.csv"
    write_calibration_matrices( outfile, C1, C2, C3 )
    print( f"Calibration matrices appended to: {outfile}." )
    
#     print( 'time to sync: %s' % ( time.time() - startTime ) )

# main

def main_test():
    e1 = np.array( [1, 0, 0] )
    e2 = np.array( [0, 1, 0] )
    root_path = "../FBG_Needle_Calibration_Data/needle_1/"
    # folder_CH1 = root_path + "12-19-19_12-32/"
    # folder_CH2 = root_path + "12-19-19_15-02/"
    # folder_CH3 = root_path + "12-19-19_15-27/"
    folder_CH1 = root_path + "12-20-19_14-04/"
    folder_CH2 = root_path + "12-20-19_15-16/"
    folder_CH3 = root_path + "12-20-19_15-43/"

    # curvature_CH1 = load_curvature(folder_CH1)
    # curvature_CH2 = load_curvature(folder_CH2)
    # curvature_CH3 = load_curvature(folder_CH3)

    # curv_vect_CH1 = get_curvature_vectors(curvature_CH1, e2)
    # curv_vect_CH2 = get_curvature_vectors(curvature_CH2, e1)
    # curv_vect_CH3 = get_curvature_vectors(curvature_CH3, -e2)
    # np.savetxt(folder_CH1 + "curvature.csv", np.hstack((curv_vect_CH1[1], curv_vect_CH1[2], curv_vect_CH1[3])))
    # np.savetxt(folder_CH2 + "curvature.csv", np.hstack((curv_vect_CH2[1], curv_vect_CH2[2], curv_vect_CH2[3])))
    # np.savetxt(folder_CH3 + "curvature.csv", np.hstack((curv_vect_CH3[1], curv_vect_CH3[2], curv_vect_CH3[3])))

    rawFBG_list1, avgFBG1, baseline1 = process_fbg(folder_CH1)
    rawFBG_list2, avgFBG2, baseline2 = process_fbg(folder_CH2)
    rawFBG_list3, avgFBG3, baseline3 = process_fbg(folder_CH3)
    np.savetxt(folder_CH1 + "rawFBGdata.csv", np.asarray(rawFBG_list1).reshape((-1,10)))
    np.savetxt(folder_CH2 + "rawFBGdata.csv", np.asarray(rawFBG_list2).reshape((-1,10)))
    np.savetxt(folder_CH3 + "rawFBGdata.csv", np.asarray(rawFBG_list3).reshape((-1,10)))

    deltaFBG_CH1 = wavelength_shift(avgFBG1, baseline1)
    deltaFBG_CH2 = wavelength_shift(avgFBG2, baseline2)
    deltaFBG_CH3 = wavelength_shift(avgFBG3, baseline3)
    np.savetxt(folder_CH1 + "wavelength_shift.csv", deltaFBG_CH1)
    np.savetxt(folder_CH2 + "wavelength_shift.csv", deltaFBG_CH2)
    np.savetxt(folder_CH3 + "wavelength_shift.csv", deltaFBG_CH3)

if __name__ == '__main__':
    # main()
    main_test()

# if

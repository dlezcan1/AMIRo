'''
Created on Dec 12, 2019

@summary: This script is intended to generate the calibration matrices relating
          curvature to FBG sensor readings
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob, re, xlrd
import time  # for debugging purposes
from datetime import timedelta, datetime
from _pickle import load
from FBGNeedle import FBGNeedle


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


def process_fbg( directory ):
    ''' Loads all FBG files, average_fbg and baseline
        rawFBG_list is a list of numpy arrays with each array holding the raw data from one file
    '''
    name_length = len( directory + "fbgdata_yyyy_mm_dd_" )
    filenames = glob.glob( directory + "fbgdata*.txt" )
    print( 'number of files: %s' % len( filenames ) )

    rawFBG = np.empty( [0, 10] )
    avgFBG = np.empty( [0, 10] )
    rawFBG_list = []

    for idx, file in enumerate( filenames ):
        baseTime = file[name_length:-4]
        hour, minute, sec = baseTime.split( '-' )
        baseInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )

        with open( file, 'r' ) as f:
            for i, line in enumerate( f ):
                if i == 0:
                    data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                    hour, minute, sec = data[0][0].split( '-' )
                    timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec )
                    offset = baseInSec - np.floor( timeInSec )
                    print( 'offset: %s' % offset )

                data = [number.strip().split( ',' ) for number in line.strip().split( ":" )]
                hour, minute, sec = data[0][0].split( '-' )
                timeInSec = float( hour ) * 3600 + float( minute ) * 60 + float( sec ) + offset

                toappend = [float( i ) for i in data[1]]  # convert to floats
                toappend.insert( 0, timeInSec )
                rawFBG = np.vstack( [rawFBG, toappend] )

        # # create baseline using first file
        if idx == 0:
            baseline = np.mean( rawFBG[:, 1:], axis = 0 )

        rawFBG_list.append( rawFBG )
        avg = np.mean( rawFBG[:, 1:], axis = 0 )
        toappend = np.hstack( [baseInSec, avg] )
        avgFBG = np.vstack( [avgFBG, toappend] )
        rawFBG = np.empty( [0, 10] )
    
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

    deltaFBG = avg_fbg[:, 1:] - baseline  # wavelength shift

    # average shift at each active area
    mean_aa1 = np.mean( deltaFBG[:, aa1_idxs], axis = 1 )
    mean_aa2 = np.mean( deltaFBG[:, aa2_idxs], axis = 1 )
    mean_aa3 = np.mean( deltaFBG[:, aa3_idxs], axis = 1 )

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


def _leastsq_fit( delta_fbg, curvature, outfile: str = '' ):
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
    
    if outfile != '':
        with open( outfile, 'w' ) as writestream:
            writestream.write( "Least Square fitting log.\n" )
            
            # AA1
            writestream.write( f"1) Residuals: {resid1}\n" )
            writestream.write( f"Relative error\nMin: {min_relerr1}\nMean: {mean_relerr1}\nMax: {max_relerr1}\n\n" )
            
            # AA2
            writestream.write( f"2) Residuals: {resid2}" )
            writestream.write( "Relative error\nMin: {min_relerr2}\nMean: {mean_relerr2}\nMax: {max_relerr2}\n\n" )
            
            # AA3
            writestream.write( f"3) Residuals: {resid3}" )
            writestream.write( f"Relative error\nMin: {min_relerr3}\nMean: {mean_relerr3}\nMax: {max_relerr3}\n\n" )
            
        # with
    # if
    
    return [np.transpose( C1 ), np.transpose( C2 ), np.transpose( C3 )]
    
# _leastsq_fit


def leastsq_fit ( dict_of_data: dict, outfile: str = None ):
    """ 
    Performs least squares fitting of the data of interest.
    
    @param dict_of_data, dict: This is a dictionary of data comprising of
            - key: 'AAX' The active area
            - data: FBG Data dict:
                - key: 'signal'/'curvature'
                - data: Data matrix of FBG signal/2-D XY curvatures
                
    @param outfile, str (Optional, Default = None): Output file path.
                
    @return: Dictionary of 'AAX' calibration matrices
     
    """
    
    if outfile != '':
        writestream = open( outfile, 'w' )
            
    # if
    
    retval = {}
    for aa, data in dict_of_data.items():
        if outfile != '':
            writestream.write( aa + ') ' )
            
        # if
        signal = data['signal']
        curvature = data['curvature']
        
        if np.float64 != signal.dtype:
            signal = np.asarray( signal, dtype = np.float64 )
            
        if curvature.dtype != np.float64:
            curvature = np.asarray( curvature, dtype = np.float64 )
        
        C, resid, rnk, sng = np.linalg.lstsq( signal, curvature, None )
        retval[aa] = C

        # run statistics
        
        delta = curvature - np.dot( signal, C )
        resid = np.linalg.norm( delta, axis = 1 )
        rel_err = delta / curvature
        rel_err[np.logical_or( rel_err == -np.inf, rel_err == np.inf )] = np.nan
        min_relerr = np.nanmin( rel_err , axis = 0 )
        mean_relerr = np.nanmean( rel_err, axis = 0 )
        max_relerr = np.nanmax( rel_err , axis = 0 )
        
        if outfile is not None:
            writestream.write( f"Residuals: {resid}\n" )
            writestream.write( f"Relative error\nMin: {min_relerr}\nMean: {mean_relerr}\nMax: {max_relerr}\n\n" )
            
        # if
        
        print( f"Residuals: {resid}\n" )
        print( f"Relative error\nMin: {min_relerr}\nMean: {mean_relerr}\nMax: {max_relerr}\n\n" )
        
    # for
    
    if outfile != '':
        writestream.close()
        print( f"Logged to file:'{outfile}'" )
    
    # if
    
    return retval
        

def plot( delta_fbg, curvature ):
    '''plots delta_fbg vs. curvature, just to see
    '''
    raise NotImplementedError( "'plot' is not implemented yet." )

# plot


def process_fbgdata_file( fbg_input_file: str, fbg_needle: FBGNeedle ):
    """ 
    This function is to handle the fbgdata.xlsx files and combine them into a single file.
    
    @param fbg_input_file, str: The fbgdata file to be processed
    
    @param fbg_needle, FBGNeedle: The FBGNeedle param class object
    
    @return: The Pandas Dataframe of the Avg. and STD for the fbg_input_file 
    """
    wb = pd.read_excel( fbg_input_file, None )  # read all sheets from wkbook
    data_sheets = [k for k in wb.keys() if k != 'Summary']  # only count the data sheets
    
    # Note: remove the end parts of the files (remove the bottom 4 when processing Avg. Std. Min. Max.)
    
    # initialize the DataFrame for all the processed data
    time_head = pd.MultiIndex.from_arrays( [['time (s)'], ['empty 2'], ['empty 3']] )
    ch_head = ['CH' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    aa_head = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    data_head = ['Average (nm)', 'STD (nm)']
    proc_header = pd.MultiIndex.from_product( [ch_head, aa_head, data_head] )
    proc_header = time_head.append( proc_header )
    
    # the empty processed data
    proc_data = pd.DataFrame( index = range( len( data_sheets ) ), columns = proc_header )
    avg_mask = proc_data.get_level_values( 2 ) == data_head[0]  # for col selection
    std_mask = proc_data.get_level_values( 2 ) == data_head[1]  # for col selection
    
    # process each frame of data from the trials to create a summary table
    for i, sheet in enumerate( data_sheets ):
        sheet_data = wb[sheet].astype( float )  # and convert all to floats
        sheet_data = sheet_data.iloc[:-4]
        
        # change the headers for better alignment
        sheet_data.columns = proc_header.droplevel( 2 ).drop_duplicates() 
        
        # add the data to the processed data
        proc_data.iloc[i, 0] = sheet_data.iloc[:, 0].min()[0]  # set the time
        proc_data.iloc[i, avg_mask] = sheet_data.iloc[:, 1:].mean()  # set the mean
        proc_data.iloc[i, std_mask] = sheet_data.iloc[:, 1:].std()  # set the STD  
    
    # for
    
    return proc_data

# process_fbgdata_file


def consolidate_fbgdata_files( fbg_input_files: list, curvature_values: list, 
                               fbg_needle: FBGNeedle, outfile: str = None ):
    """
    This function is used to consolidate the FBGdata file lists 
    
    @param fbg_input_files, list: list of fbgdata.xlsx files to be processed.
    
    @param curvature_values, list: list of associated curvatures induced in
                fbgdata.xlsx files list.
                
    @param fbg_needle, FBGNeedle: the FBGNeedle class param object
    
    @param outfile, str (Optional, Default = None): Output file path. If is 'None',
                then no file will be saved.

    @return: The entire pandas Dataframe consolidated with curvature and processed
                fbgdata.xlsx averages and std.
    """
    # data checking
    if len( curvature_values ) != len( fbg_input_files ):
        raise IndexError( "The curvature values and fbg files must be of the same length." )
    
    # initialize the array with the first sheet
    first_head = pd.MultiIndex.from_arrays( [['Curvature (1/m)', 'time (s)'],
                                            2 * ['empty 2'], 2 * ['empty 3']] )
    ch_head = ['CH' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    aa_head = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    data_head = ['Average (nm)', 'STD (nm)']
    
    all_data_header = pd.MultiIndex.from_product( [ch_head, aa_head, data_head] )
    all_data_header = first_head.append( all_data_header )
    
    all_data = pd.DataFrame( columns = all_data_header )
    
    # Begin collecting data from the fbgdata.xlsx files
    for fbg_file, curvature in zip( fbg_input_files, curvature_values ):
        # run the curvatures
        df = process_fbgdata_file( fbg_file, fbg_needle )  # get the processsed df
        curv_ds = pd.Series( curvature * np.ones( df.shape[0] ), name = 'Curvature (1/m)' ) 
        
        # concatenate the data
        all_data = all_data.append( pd.concat( [curv_ds, df], axis = 1 ), ignore_index = True )
        
    # for
    
    # save the data
    if outfile is not None:
        all_data.to_excel( outfile )
        
    return all_data
    
# consolidate_fbgdata_files


def create_datamatrices( fbg_input_files: list, fbg_needle: FBGNeedle,
                          outfile: str = None ):
    """ 
    This function is used to consolidate the data into a single
    data matrix for multiple AA from the compiled data. 
    """
    
    raise NotImplementedError( "'create_datamatrices_file' is not implemented yet." )

# create_datamatrices_file


def read_datamatrices( filename: str, fbg_needle: FBGNeedle ):
    """ To read the excel data matrices summary file """
    wkbook = xlrd.open_workbook( filename )
    
#     data_areas = ['AA' + str( i + 1 ) for i in range( num_active_areas )]
    data_areas = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    
    retval = {}  # dictionary to return of dictionaries
    for area in data_areas:
        area_sheet = wkbook.sheet_by_name( area )
        rows = area_sheet.get_rows()
        
        curvature = np.empty( ( 0, 2 ) , dtype = float )
#         fbg_signal = np.empty( ( 0, 3 ) , dtype = float )
        fbg_signal = np.empty( ( 0, fbg_needle.num_channels ) , dtype = float )
        
        curvature_read = False
        fbg_read = False
        for row in rows:
            if row[0].value == "Curvature:" and not curvature_read:
                curvature_read = True
                fbg_read = False
            
            # if
            
            elif row[0].value == "Signal:" and not fbg_read:
                fbg_read = True
                curvature_read = False
            
            # if
            
            if curvature_read:
                data = [d.value for d in row[:2]]
                if isinstance( data[0], float ):
                    curvature = np.vstack( ( curvature, data ) )
                
            # if
            
            if fbg_read:
                data = [d.value for d in row[:3]]
                if isinstance( data[0], float ):
                    fbg_signal = np.vstack( ( fbg_signal, data ) )
                
            # if
            
        # for
        
        retval[area] = {'curvature': curvature, 'signal': fbg_signal}
        
    # for
    
    return retval

# read_data_matrices


def write_calibration_matrices( outfile: str, C_list: list ):
    """ Appends the calibration matrices to a file """
    with open( outfile, 'a' ) as writestream:
        for aa, C in C_list.items():
            msg = np.array2string( C.T, separator = ',' ).replace( '[', '' ).replace( ']', '' )
            
            writestream.write( aa + ":\n" )  # write AAi:
            writestream.write( msg + '\n\n' )  # write the calibration matrix
            
        # for
        
    # with
    
    return 0

# write_calibration_matrices


def _write_calibration_matrices( outfile, C1, C2, C3 ):
    """ DEPRECATED
    Method to simplify writing the calibration matrices to a file
    """
    with open( outfile, 'a' ) as writestream:
        # format the matrices
        msg1 = np.array2string( C1, separator = ',' ).replace( '[', '' ).replace( ']', '' )
        msg2 = np.array2string( C2, separator = ',' ).replace( '[', '' ).replace( ']', '' )
        msg3 = np.array2string( C3, separator = ',' ).replace( '[', '' ).replace( ']', '' )
        
        # write the data
        writestream.write( "AA1:\n" + msg1 + '\n\n' )
        writestream.write( "AA2:\n" + msg2 + '\n\n' )
        writestream.write( "AA3:\n" + msg3 + '\n\n' )
        
    # with
    
    return 0

# _write_calibration_matrices


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
    
    C1, C2, C3 = _leastsq_fit( delta_fbg, curv_vect )
#     return;
    outfile = root_path + "needle_params.csv"
    _write_calibration_matrices( outfile, C1, C2, C3 )
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

    rawFBG_list1, avgFBG1, baseline1 = process_fbg( folder_CH1 )
    rawFBG_list2, avgFBG2, baseline2 = process_fbg( folder_CH2 )
    rawFBG_list3, avgFBG3, baseline3 = process_fbg( folder_CH3 )
    np.savetxt( folder_CH1 + "rawFBGdata.csv", np.asarray( rawFBG_list1 ).reshape( ( -1, 10 ) ) )
    np.savetxt( folder_CH2 + "rawFBGdata.csv", np.asarray( rawFBG_list2 ).reshape( ( -1, 10 ) ) )
    np.savetxt( folder_CH3 + "rawFBGdata.csv", np.asarray( rawFBG_list3 ).reshape( ( -1, 10 ) ) )

    deltaFBG_CH1 = wavelength_shift( avgFBG1, baseline1 )
    deltaFBG_CH2 = wavelength_shift( avgFBG2, baseline2 )
    deltaFBG_CH3 = wavelength_shift( avgFBG3, baseline3 )
    np.savetxt( folder_CH1 + "wavelength_shift.csv", deltaFBG_CH1 )
    np.savetxt( folder_CH2 + "wavelength_shift.csv", deltaFBG_CH2 )
    np.savetxt( folder_CH3 + "wavelength_shift.csv", deltaFBG_CH3 )
    
# main_test


def main_calmat():
    directory = "../FBG_Needle_Calibration_Data/needle_1/"
    datadir = directory + "Jig_Calibration/"
    datafile = "Data Matrices.xlsx"
    needleparamfile = "needle_params.csv"
    needlejsonfile = "needle_params.json"
    out_needlejsonfile = needlejsonfile[:-5] + '-' + datadir.split('/')[-2] + '.json'
    lstsq_logfile = "least_sq.log"

    fbg_needle = FBGNeedle.load_json( directory + needlejsonfile )
    calibration_data = read_datamatrices( datadir + datafile, fbg_needle )
    
#     # check if data is being read in correctly
#     for a in calibration_data.keys():
#         print( a )
#         print( "curvature:" )
#         print( calibration_data[a]['curvature'], '\n' )
#         print( "signal" )
#         print( calibration_data[a]['signal'], '\n' )
#         print( 75 * '=' )
#         
#     # for

    calibration_matrices = leastsq_fit( calibration_data, directory + lstsq_logfile )
    write_calibration_matrices( datadir + needleparamfile, calibration_matrices )
    print( f"Wrote calibration matrices to '{needleparamfile}'" )
    
# main_calmat


if __name__ == '__main__':
    # main()
#     main_test()
    main_calmat()
    print( "Program Terminated." )

# if

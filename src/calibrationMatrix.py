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


def _read_datamatrices( filename: str, fbg_needle: FBGNeedle ):
    """ DEPRECATED
    To read the excel data matrices summary file
     """
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

# _read_data_matrices


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


def create_datamatrices( fbgresult_files: dict, fbg_needle: FBGNeedle,
                          outfile: str = None ):
    """ 
    This function is used to consolidate the data into a single
    data matrix for multiple AA from the compiled data. 
    """
    
#     raise NotImplementedError( "'create_datamatrices' is not implemented yet." )
    unit_vec = lambda theta: np.array( [[np.cos( np.deg2rad( theta ) - np.pi / 2 ), np.sin( np.deg2rad( theta ) - np.pi / 2 )]] )
    
    # create the total data dict of pandas datasheets
    ch_head = ['CH' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    curv_head = ['Curvature x', 'Curvature y', 'Curvature']
    data_total = {}
    for aa in range( fbg_needle.num_aa ):
        data_total[aa + 1] = pd.DataFrame( columns = curv_head + ch_head )
        
    # for 
    
    # load the data from all of the files
    for angle, fbgresult in fbgresult_files.items():
        # the direction of curvature
        curv_unit_vector = unit_vec( angle ) 
        
        # load the data from the FBGresult file
        data = pd.read_excel( fbgresult, sheet_name = 'Data Summary', header = [0, 1] )
        
        # load the curvature values
        k = data.loc[data['Curvature'].iloc[:, 0].isna() == False, 'Curvature'].iloc[:, 0:fbg_needle.num_aa]
        
        # load the active area data and perform T compensation
        col_list = ['Curvature'] + ['Ch ' + str( i + 1 ) for i in range( fbg_needle.num_channels )]  # for data ref'ing
        for aa in range( fbg_needle.num_aa ):
            # not T corrected
            data_aai_tmp = data['Active Area {:d}'.format( aa + 1 )][col_list].iloc[:k.shape[0]]
            curv_data_tbl = data_aai_tmp[['Curvature']].dot( curv_unit_vector ).rename( lambda i: curv_head[i],
                                                                                        axis = 1 )  # curvature vectorization
            curv_data_tbl = curv_data_tbl.astype( float ).round( 2 )  # remove rounding erros
            data_aai_tmp = curv_data_tbl.join( data_aai_tmp )  # add the curvature there
            
            # perform the T correction
            mean_wl = data_aai_tmp[col_list[1:]].mean( 1 )
            data_aai_tmp[col_list[1:]] = data_aai_tmp[col_list[1:]].subtract( mean_wl, axis = 0 ) 
            
            # reformat the header
            data_aai_tmp.columns = data_aai_tmp.columns.str.replace( 'Ch ', 'CH' )
            
            # append the data to the total table (per AA)
            data_total[aa + 1] = data_total[aa + 1].append( data_aai_tmp, ignore_index = True )
            
        # for
    
    # for
    
    # write the outfile
    if outfile:
        xl_writer = pd.ExcelWriter( outfile, engine = 'xlsxwriter' )  # the Excel writer
        
        # write each AA to it's header file
        for aa, data in data_total.items():
            data.to_excel( xl_writer, sheet_name = "AA" + str( aa ) )
            
        # for
        
        xl_writer.save()  # write and close the Excel file
        
    # if
    
    return data_total

# create_datamatrices_file


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


def leastsq_fit ( dict_of_data: dict, outfile: str = None, curv_wgt_rule = lambda k: 1 )    :
    """ 
    Performs least squares fitting of the data of interest.
    
    @param dict_of_data, dict: This is a dictionary of data comprising of
            - key: 'AAX' The active area
            - data: FBG Data dict:
                - key: 'signal'/'curvature'
                - data: Data matrix of FBG signal/2-D XY curvatures
                
    @param outfile, str (Optional, Default = None): Output file path.
    
    @param curv_wgt_rule(Optional, Default = 1 for all weights, function to provide
             weighting of lst squares based on curvature 
                
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
        
        # perform the weighting
        weights = np.sqrt( np.diag( [curv_wgt_rule( k ) for k in np.linalg.norm( curvature, axis = 1 )] ) )
        signal_w = weights.dot( signal )
        curvature_w = weights.dot( curvature )
        
        C, resid, rnk, sng = np.linalg.lstsq( signal_w, curvature_w, None )
        retval[aa] = C.T

        # run statistics
        
        delta = curvature - np.dot( signal, C )
        resid = np.linalg.norm( delta, axis = 1 )
        rel_err = delta / curvature
        rel_err[np.logical_or( rel_err == -np.inf, rel_err == np.inf )] = np.nan
        min_relerr = np.nanmin( rel_err , axis = 0 )
        mean_relerr = np.nanmean( np.abs( rel_err ), axis = 0 )
        max_relerr = np.nanmax( rel_err , axis = 0 )
        
        if outfile is not None:
            writestream.write( f"Residuals: {resid}\n" )
            writestream.write( f"Relative error\nMin: {min_relerr}\nMean: {mean_relerr}\nMax: {max_relerr}\n\n" )
            
        # if
    
        print( aa )
        print( f"Residuals: {resid}\n" )
        print( f"Relative error\nMin: {min_relerr}\nMean: {mean_relerr}\nMax: {max_relerr}\n\n" )
        print( 75 * '=' )
        
    # for
    
    if outfile != '':
        writestream.close()
        print( f"Logged to file:'{outfile}'" )
    
    # if
    
    return retval

# leastsq_fit


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


def load_filteredfbg_data( filename: str ):
    """ Method to read in the fbg filtered data """
    inp = np.loadtxt( filename, np.float64 )
    
    ts = inp[:, 0]
    data = inp[:, 1:]
    
    return ts, data

# load_filteredfbg_data


def perform_validation( valid_data_file: str, fbg_needle: FBGNeedle, out_fmt = "dict" ):
    """ This is to perform the validation given a calibrated FBGNeedle """
    # valid arguments
    valid_out_fmt_args = ["dict", "pandas"]
    
    # check input arguments for validity
    if not out_fmt in valid_out_fmt_args:
        raise ValueError( "'{:s}' is not a valid out_fmt argument.".format( out_fmt ) )
    
    # if
    
    # validation data matrices
    valid_data = read_datamatrices( valid_data_file, fbg_needle )
    
    # iterate through each AA to see projected calibration
    retval = {}
    for key, data_aa in valid_data.items():
        # load the calibration matrices and data
        C_aa = fbg_needle.aa_cal( key ).T  # calib. matrix (transpose for row-based)
        signal = data_aa['signal']  # fbg signal
        curv_exp = data_aa['curvature']  # curvature
        
        # perform the fit
        curv_pred = signal.dot( C_aa )
        
        # append the data expected vs. predicted data
        if out_fmt == valid_out_fmt_args[0]:
            retval[key] = {'expected': curv_exp, 'predicted': curv_pred}
        
        # if        
        elif out_fmt == valid_out_fmt_args[1]:
            tbl_head = pd.MultiIndex.from_product( [['Expected', 'Predicted'],
                                                   ['Curvature x', 'Curvature y']] )
            tbl_data = np.hstack( ( curv_exp, curv_pred ) )
            tbl = pd.DataFrame( tbl_data , columns = tbl_head )
            
            retval[key] = tbl
            
        # elif
        
    # for
    
    return retval
    
# perform_validation


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


def read_datamatrices( filename: str, fbg_needle: FBGNeedle ):
    """ This is a function to read the new 'Data Matices.xlsx' file as pandas table"""    
#     raise NotImplementedError( "'read_datamatrices' is not yet implemented." )
    sheets = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    curv_head = ['Curvature x', 'Curvature y']
    ch_head = ['CH' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    
    retval = {}
    # process each AA sheet in the Data Matrices file 
    for sheet_aa in sheets:
        # read in the data sheet
        data_aa = pd.read_excel( filename, sheet_name = sheet_aa, index_col = 0 )
        
        # get the numpy array so f the curvature and signal values
        curvature = data_aa[curv_head].values
        signal = data_aa[ch_head].values
        
        # append the data to the retval dict
        retval[sheet_aa] = {'curvature': curvature, 'signal': signal}
        
    # for
    
    return retval

# read_datamatrices


def read_FBGResult_summary( file: str, fbg_needle: FBGNeedle ):
    """ This is to read in the data summary from the FBGResults Excel workbook """
    
    raise NotImplementedError( "'read_FBGResult_summary' is not implemented yet." )

    # preparation
    aa_head = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    ch_head = ['Ch ' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    
    sheet = pd.read_excel( file, sheet_name = 'Data Summary', header = [0, 1] )
    
    # get the curvature values
    curvature = sheet.loc[sheet['Curvature'].iloc[:, 0].isna() == False, 'Curvature'].iloc[:fbg_needle.num_aa]
    curvature.columns = pd.MultiIndex.from_product( [['Curvature', aa_head]] )
    
    # get the AA data
    aa_data = {}
    for i in range( fbg_needle.num_aa ):
        aa_data_i = {}
        
        # collect the no T corrected data
        aa_data_i['No T'] = sheet['Active Area {:d}'.format( i + 1 )].iloc[:curvature.shape[0],
                                                                                  :-2].set_index( 'Curvature' )
        aa_data_i['T'] = aa_data_i.sub( aa_data_i.mean( 1 ), axis = 0 )  # subtract mean
        
        aa_data[i + 1] = aa_data_i
        
    # for
    
    return aa_data
    
# read_FBGResult_summary


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


def write_calibration_matrices( C_list: dict, fbg_needle: FBGNeedle, outfile: str = None, fbg_outjson_file: str = None ):
    """ Appends the calibration matrices to a file """
    if outfile:
        with open( outfile, 'a' ) as writestream:
            for aa, C in C_list.items():
                C_list[aa] = C
                msg = np.array2string( C, separator = ',' ).replace( '[', '' ).replace( ']', '' )
                
                writestream.write( aa + ":\n" )  # write AAi:
                writestream.write( msg + '\n\n' )  # write the calibration matrix
                
            # for    
        # with
    # if
    
    if fbg_outjson_file:
        fbg_needle.cal_matrices = C_list
        fbg_needle.save_json( fbg_outjson_file )
    # if
    
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
    directory = "../FBG_Needle_Calibration_Data/needle_3CH_4AA/"
    needlejsonfile = "needle_params.json"
    
    datadir = directory + "Jig_Calibration_08-05-20/"
    datafile = "Data Matrices.xlsx"
    needleparamfile = "needle_params.csv"
    out_needlejsonfile = needlejsonfile[:-5] + '-' + datadir.split( '/' )[-2] + '.json'
    
    lstsq_logfile = "least_sq.log"

    fbg_needle = FBGNeedle.load_json( directory + needlejsonfile )
    calibration_data = _read_datamatrices( datadir + datafile, fbg_needle )
    
    # check if data is being read in correctly
    for a in []:  # calibration_data.keys():
        print( a )
        print( "curvature:" )
        print( calibration_data[a]['curvature'], '\n' )
        print( "signal" )
        print( calibration_data[a]['signal'], '\n' )
        print( 75 * '=' )
         
    # for

    calibration_matrices = leastsq_fit( calibration_data, directory + lstsq_logfile )
    write_calibration_matrices( calibration_matrices, fbg_needle,
                                fbg_outjson_file = directory + out_needlejsonfile )
    print( f"Wrote calibration matrices to '{needleparamfile}'" )
    
# main_calmat


def main_dbg():
    directory = "../FBG_Needle_Calibration_Data/needle_3CH_4AA/"
    needlejsonfile = "needle_params.json"
    
    datadir = directory + "Jig_Calibration_08-05-20/"
    datadir = directory + "Validation_Temperature_08-12-20/"
    datafile = "Data Matrices.xlsx"
    
    # test the create_datamatrices function
    fbg_needle = FBGNeedle.load_json( directory + needlejsonfile )
    fbgresult_list = glob.glob( datadir + "*JigValidation-Temperature_results*.xlsx" )
    fbgresult_files = {0: fbgresult_list[0], 90: fbgresult_list[-1]}
    
    create_datamatrices( fbgresult_files, fbg_needle, datadir + datafile )
    print( "Saved file: ", datadir + datafile )

# main_dbg


def main_validation():
    # load the file information needed
    directory = "../FBG_Needle_Calibration_Data/needle_3CH_4AA/"
    needlejsonfile = "needle_params-Jig_Calibration_08-05-20.json"
    
    datadir = directory + "Validation_Temperature_08-12-20/"
    datafile = "Data Matrices.xlsx"
    
    out_file = datadir + "Validation_Error.xlsx"
    
    # load the json file
    fbg_needle = FBGNeedle.load_json( directory + needlejsonfile )
    
    # get the validation-prediction data
    valid_pred = perform_validation( datadir + datafile, fbg_needle, out_fmt = 'pandas' )
    
    # perform the analysis
    for key, prediction_aa in valid_pred.items():
        # pred - exp 
        dev = prediction_aa['Expected'] - prediction_aa['Predicted']  # deviation
        prediction_aa[[('Error', 'Curvature x'), ('Error','Curvature y')]] = dev
        
        # | norm(pred) - norm(exp) | amount of curvature error 
        norms = np.vstack( prediction_aa.groupby( axis = 1, level = 0 ).apply( np.linalg.norm, axis = 1 ) )
        prediction_aa[( 'Error', 'norm (1/m)' )] = np.abs( norms[0] - norms[1] ).T
        prediction_aa[( 'Error', 'rel. norm' )] = prediction_aa[( 'Error', 'norm (1/m)' )] / norms[0]
        
        # norm(pred - exp) L2 
        prediction_aa[( 'Error', 'L2 (1/m)' )] = np.linalg.norm( dev , axis = 1 ) 
        prediction_aa[( 'Error', 'rel. L2' )] = prediction_aa[( 'Error', 'L2 (1/m)' )] / norms[0] 
        
        # arg( pred - exp ) angular deviation
        prediction_aa[( 'Error', 'Arg (rads)' )] = np.arctan2( dev['Curvature y'], dev['Curvature x'] )
        prediction_aa[( 'Error', 'Arg (degs)' )] = np.rad2deg( prediction_aa[( 'Error', 'Arg (rads)' )] )
        
        # change the data table
#         valid_pred[key] = prediction_aa
        
    # for
    
    # write the processed data
    xlwriter = pd.ExcelWriter( out_file, engine = 'xlsxwriter' )
    for key, prediction_aa in valid_pred.items():
        prediction_aa.to_excel( xlwriter, sheet_name = key )
        
    # for
    
    xlwriter.save()
    print( "Saved file:", out_file )
    
# main_validation


if __name__ == '__main__':
    # main()
#     main_test()
#     main_calmat()
#     main_dbg()
    main_validation()
    print( "Program Terminated." )

# if

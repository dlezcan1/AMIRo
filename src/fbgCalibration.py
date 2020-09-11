'''
Created on Dec 6, 2019

@author: Dimitri Lezcano

@summary: This script is intended for the use of performing the data processing
          incorporating the image and FBG data when needle calibration is 
          performed.
'''
import glob, re, cv2, xlsxwriter, os.path, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import calibrationMatrix as calmat
import image_processing as imgp
from datetime import datetime, timedelta
from scipy import interpolate
from FBGNeedle import FBGNeedle
# from needle_segmentation_script import CROP_AREA

# the time formatting of the data
TIME_FMT = "%H-%M-%S.%f"
TIME_FIX = timedelta( hours = 0 )  # the internal fix for the time

# Image processing params
# PIX_PER_MM = 8.875 # what appears to be the actual fit
PIX_PER_MM = 8.498439767625596
# CROP_AREA = ( 32, 425, 1180, 600 )
CROP_AREA = ( 40, 420, 1230, 620 )
BO_REGIONS = [( 0, 0, -1, 30 ), ( 1060, 125, -1, -1 )]
BO_REGIONS.append( ( 1090, 105, -1, -1 ) )
BO_REGIONS.append( ( 0, 75, 145, -1 ) )
BO_REGIONS.append( ( 950, 0, -1, 68 ) )
BO_REGIONS.append( ( 0, 170, -1, -1 ) )


def calculate_shift_stepped( summ_data: pd.DataFrame, fbg_needle: FBGNeedle, Tcorr: bool = True ):
    """ This function is to calculate the (T corrected) signal shift """
    # get the CH's and AA's
    aa_list = summ_data.columns.unique( 1 )
    
    # the reference wavelength
    ref_wl = summ_data.loc[0, pd.IndexSlice[:, :, 'Average (nm)']].iloc[0]
    
    # calculate the wavelength shift
    shift_mean_data = summ_data.loc[:, pd.IndexSlice[:, :, 'Average (nm)']].subtract( ref_wl, axis = 1 )
    shift_std_data = summ_data.loc[:, pd.IndexSlice[:, :, 'STD (nm)']]
        
    if Tcorr:  # perform temperature compensation (through averaging
        mean_aa_shift = shift_mean_data.droplevel( 2, axis = 1 ).groupby( axis = 1, level = 1 ).mean()
        # subtract the mean AA shift for each row
        for aa in aa_list:
            shift_mean_data.loc[:, pd.IndexSlice[:, aa, 'Average (nm)']] = shift_mean_data.loc[:, pd.IndexSlice[:, aa, 'Average (nm)']].subtract( mean_aa_shift.loc[:, aa], axis = 0 )
            
        # for
    # if
    
    return shift_mean_data.join( shift_std_data ).sort_index( axis = 1, level = 0 )
    
# calculate_shift_stepped


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
        curv_ds = pd.Series( curvature * np.ones( df.shape[0] ), name = first_head[0] ) 
        
        # concatenate the data
        all_data = all_data.append( pd.concat( [curv_ds, df], axis = 1 ), ignore_index = True )
        
    # for
    all_data = all_data.astype( float )
    
    # summarize the trials by curvature
    # create the summary table
    summ_header = all_data.columns[2:]
    summ_data = pd.DataFrame( index = curvature_values, columns = summ_header )
    summ_data.index.name = "Curvature (1/m)"
    
    # get the AVG and STD for the wavalength values for each trial
    mask_avg = np.logical_or( all_data.columns.get_level_values( 2 ) == data_head[0],
                             all_data.columns.get_level_values( 0 ) == "Curvature (1/m)" )
    summ_mean = all_data.iloc[:, mask_avg].groupby( by = first_head[0] ).mean()  # mean values
    summ_std = all_data.iloc[:, mask_avg].groupby( by = first_head[0] ).std()  # std values
    
    # find the AVG and STD masks in the summary data table
    mask_avg = summ_data.columns.get_level_values( 2 ) == data_head[0]
    mask_std = summ_data.columns.get_level_values( 2 ) == data_head[1]
    
    # make sure the headers are set correct in head table
    summ_mean.columns = summ_data.columns[mask_avg]
    summ_std.columns = summ_data.columns[mask_std]
    
    # add the AVG and STD values to the table
    summ_data.iloc[:, mask_avg] = summ_mean
    summ_data.iloc[:, mask_std] = summ_std
    
    # save the data
    if outfile is not None:
        xlwriter = pd.ExcelWriter( outfile, engine = 'xlsxwriter' )
        summ_data.to_excel( xlwriter, sheet_name = 'Summary' )
        all_data.to_excel( xlwriter, sheet_name = 'Trial Data' )
        
        xlwriter.save()
        
    # if
        
    return summ_data, all_data
    
# consolidate_fbgdata_files


def consolidate_fbgdata_files_stepped( fbg_input_files: list, curvature_values: list,
                                      step_trials: list, ref_trials: int,
                                      fbg_needle: FBGNeedle, outfile: str = None ):
    """
    This function is used to consolidate the FBGdata file lists for progressive insertion
    
    @param fbg_input_files, list: list of fbgdata.xlsx files to be processed.
    
    @param curvature_values, list: list of associated curvatures induced in
                fbgdata.xlsx files list.
                
    @param step_trials, list: list of insertion length for stepped insertion
    
    @param ref_trials, int: the number of reference trials taken
    
    @param fbg_needle, FBGNeedle: the FBGNeedle class param object
    
    @param outfile, str (Optional, Default = None): Output file path. If is 'None',
                then no file will be saved.

    @return:    - The summary over all trials of the data with curvature and processed
                    fbgdata.xlsx averages and std.
                - the summary for each trial of the data with curvature and processed
                    fbgdata.xlsx averages and std.
                -The entire pandas Dataframe consolidated with curvature and processed
                    fbgdata.xlsx averages and std.
    """
    # data checking
    if len( curvature_values ) != len( fbg_input_files ):
        raise IndexError( "The curvature values and fbg files must be of the same length." )
    
    # reg-exp for directory
    pattern = re.compile( ".*/([0-9]+)_deg/Trial_([0-9]+)/.*/*.xlsx" )  # used to get expmnt. angle & trial #
    pattern = re.compile( ".*/Trial_([0-9]+)/.*/.*\.xlsx" )
    
    # initialize the array with the first sheet
    first_head_arr = ['Trial', 'Curvature (1/m)', 'time (s)', 'Insertion Length (mm)']
    first_head = pd.MultiIndex.from_arrays( [first_head_arr,
                                            len( first_head_arr ) * ['empty 2'],
                                            len( first_head_arr ) * ['empty 3']] )
    ch_head = ['CH' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    aa_head = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    data_head = ['Average (nm)', 'STD (nm)']
        
    all_data_header = pd.MultiIndex.from_product( [ch_head, aa_head, data_head] )
    all_data_header = first_head.append( all_data_header )
    
    all_data = pd.DataFrame( columns = all_data_header )  # all data sheet
    mask_avg = all_data.columns.get_level_values( 2 ) == data_head[0]
    mask_std = all_data.columns.get_level_values( 2 ) == data_head[1]
    
    # Begin collecting data from the fbgdata.xlsx files
    for fbg_file, curvature in zip( fbg_input_files, curvature_values ):
        # run the curvatures
        trial_num = int( re.match( pattern, fbg_file ).group( 1 ) )
        df = process_fbgdata_file_stepped( fbg_file, step_trials, ref_trials, fbg_needle ).astype( float )  # get the processsed df
        trial_ds = pd.Series( trial_num * np.ones( df.shape[0] ), name = first_head[0] ).astype( int )
        curv_ds = pd.Series( curvature * np.ones( df.shape[0] ), name = first_head[1] ).astype( float )
        
        # concatenate the data
        all_data = all_data.append( pd.concat( [trial_ds, curv_ds, df], axis = 1 ), ignore_index = True )
        
    # for
    all_data = all_data[all_data_header].convert_dtypes()
    
    # temp table for processing
    tmp_tbl = all_data.drop( labels = [first_head_arr[2]], axis = 1, level = 0 ).drop( labels = ['STD (nm)'], axis = 1, level = 2 )  # all data w/o time & STD
    
    # summarize the trials by curvature 
    # avg and std for each trial
    summ_trial_avg_data = tmp_tbl.groupby( [first_head[1], first_head[3], first_head[0]] ).mean()
    summ_trial_std_data = tmp_tbl.groupby( [first_head[1], first_head[3], first_head[0]] ).std()
    summ_trial_avg_data.columns = all_data.columns[mask_avg]
    summ_trial_std_data.columns = all_data.columns[mask_std]
    summ_trial_data = summ_trial_avg_data.join( summ_trial_std_data )
    summ_trial_data.index.names = [indx[0] for indx in summ_trial_data.index.names]
    summ_trial_data = summ_trial_data.reindex( columns = sorted( summ_trial_data.columns ), copy = False )
    
    # avg and std over all trials (curvature | insertion length)
    summ_avg_data = tmp_tbl.drop( labels = [first_head_arr[0]], axis = 1, level = 0 ).groupby( [first_head[1], first_head[3]] ).mean()
    summ_std_data = tmp_tbl.drop( labels = [first_head_arr[0]], axis = 1, level = 0 ).groupby( [first_head[1], first_head[3]] ).std()
    summ_avg_data.columns = all_data.columns[mask_avg]
    summ_std_data.columns = all_data.columns[mask_std]
    summ_data = summ_avg_data.join( summ_std_data )
    summ_data.index.names = [indx[0] for indx in summ_data.index.names]
    summ_data = summ_data.reindex( columns = sorted( summ_data.columns ), copy = False )
    
    summ_data_flip = summ_data.reorder_levels( [1, 0], axis = 0 )
    
    # save the data
    if outfile is not None:
        xlwriter = pd.ExcelWriter( outfile, engine = 'xlsxwriter' )
        summ_data.to_excel( xlwriter, sheet_name = 'Summary' )
        summ_data_flip.to_excel( xlwriter, sheet_name = 'Summary Reindexed' )
        summ_trial_data.to_excel( xlwriter, sheet_name = 'Summary Trial' )
        all_data.to_excel( xlwriter, sheet_name = 'Trial Data' )
        
        xlwriter.save()
        
    # if
        
    return summ_data, summ_data_flip, summ_trial_data, all_data
    
# consolidate_fbgdata_files_stepped


def load_curvature( directory: str, filefmt: str = "curvature_monofbg*.txt" ):
    """ Function to load the curvature data. """
    global  TIME_FMT
    curv_files = glob.glob( directory + filefmt )
    timefmt = "%m-%d-%Y_" + TIME_FMT
    pattern = "([0-9]+)-([0-9]+)-([0-9]+)_([0-9]+)-([0-9]+)-([0-9]+).([0-9]+).txt"
    
    curv_data = np.empty( ( 0, 4 ) )
    
    for file in curv_files:
        data = np.empty( 0 )
        mon, day, yr, hr, minutes, sec, ns = re.search( pattern, file ).groups()
        ms = ns[:-3]
        ts_str = '-'.join( [mon, day, yr] ) + '_' + '-'.join( [hr, minutes, sec] ) + "." + ms
        ts = datetime.strptime( ts_str, timefmt )
        t0 = datetime( ts.year, ts.month, ts.day ) 
        dt = ( ts - t0 ).total_seconds()
        data = np.append( data, dt )
        
        with open( file, 'r' ) as readstream:
            for line in readstream:
                s = line.split( ':' )
                
                if len( s ) != 2:
                    continue
                
                else:
                    desc, d = s
                    
                if desc == "Curvatures (1/mm)":
                    curvature = np.fromstring( d, sep = ',' , dtype = float )
                    data = np.append( data, curvature )
                    break
                
                # if
            # for
        # with
        
        curv_data = np.vstack( ( curv_data, data ) )
        
    # for
    
    curv_data = curv_data[curv_data[:, 0].argsort()]
    return curv_data

# load_curvature


def fix_fbgData( filename: str ):
    """ function to fix the fbg data formatting (remove unnecessary line breaks)"""

    timestamps = np.empty( 0 )
    fbgdata = np.empty( 0 )
    
    with open( filename, 'r' ) as file:
        lines = file.read().split( ":" )
        
    # with
    
    new_file = filename.split( '/' )
    new_file[-1] = "fixed_" + new_file[-1]
    new_file = '/'.join( new_file )
    
    with open( new_file, 'w+' ) as writestream:
        for i, line in enumerate( lines ):
#             print( 100 * i / len( lines ), '%' )
            
            d = line.split( ' ' )
            if len( d ) == 1 and i < len( lines ) - 1:  # not at the end
                timestamps = np.append( timestamps, line + ": " )
                
            elif len( d ) == 1 and i >= len( lines ) - 1:  # at the end
                fbgdata = np.append( fbgdata, line.replace( '\n', '' ) )
            
            else:
                ts = line.split( '\n' )[-1] + ':'
                data = line.split( '\n' )[:-1]
                data = " ".join( data ).replace( '\n', '' )
#                 data = " ".join( data.split( '\n' ) )
                timestamps = np.append( timestamps, ts )
                fbgdata = np.append( fbgdata, data )
                
            # else
            
            if len( timestamps ) > 0 and len( fbgdata ) > 0:
                ts = timestamps[0]
                timestamps = timestamps[1:]
                d = fbgdata[0]
                fbgdata = fbgdata[1:]
                
                writestream.write( ts + d + '\n' )
                
            # if
            
    # with

    print( "Wrote file:", new_file )
#     with open( new_file, 'w+' ) as file:
#             for ts, d in zip( timestamps, fbgdata ):
#                 file.write( ts + d + '\n' );
#                 
#     # with
        
# fix_fbgData


def get_curvature_image ( filename: str, active_areas: np.ndarray, needle_length: float,
                          display: bool = False, polfit: int = 5 ):
    """ Method to get curvature @ the active areas & output to data file."""
    global PIX_PER_MM, CROP_AREA, BO_REGIONS
    
    smooth_iter = 1 
    smooth_win = 25  # px
    curv_dx = 0.5  # px
    circ_win = 10  # mm
    poly_fit = polfit
    
    imgpconfig = '\n'.join( ( "Configuatation:",
           f"Curvature Determination Type: Circle fitting to polynomial",
           f"Circle Fitting Window: {circ_win} mm",
           f"Curvature interpolation size: {curv_dx}px",
           f"Smoothing Window size: {smooth_win} px",
           f"Smoothing iterations: {smooth_iter}"
           ) )
    
    print( 75 * '=' )
    print( imgpconfig )
    print( 75 * '=' , '\n' )
    
    img, gray_img = imgp.load_image( filename )
    
    gray_img = imgp.saturate_img( gray_img, 1.4, 14 )
    crop_img = imgp.set_ROI_box( gray_img, CROP_AREA )
    canny_img = imgp.canny_edge_detection( crop_img, display, BO_REGIONS )
    
    skeleton = imgp.get_centerline ( canny_img )
    if display:
        cv2.imshow( "Skeleton", skeleton )
    
    # if
    
    if display:
        cv2.waitKey( 0 )
        cv2.destroyAllWindows()
    
    # if
    
    poly, x = imgp.fit_polynomial( skeleton, poly_fit )
    xc, yc = imgp.xycenterline( skeleton )
    x = np.sort( x )  # sort the x's  (just in case)
    x0 = x.max()  # start integrating from the tip
    lb = x.min()
    ub = x.max()
    x_active = imgp.find_active_areas( x0, poly, active_areas, PIX_PER_MM, lb, ub )
    x_active = np.array( x_active ).reshape( -1 )
    
    # set-up figures
    fig, axs = plt.subplots( 2 )
    fig.suptitle( filename )
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    # plot the polynomial on top of the image
    imgp.plot_func_image( crop_img, poly, x, axs[0] )
    
    # plot active areas
    for x_a in x_active:
        axs[0].axvline( x = x_a, color = 'r' )
        axs[1].axvline( x = x_a, color = 'r' )
        
    # for
    
    # get the curvature along the image using polynomial
    curv_plot = imgp.fit_circle_curvature( poly, x, x, circ_win * PIX_PER_MM, dx = curv_dx )
    
    # get the curvature along the image using raw xy-centerline
#     curv_plot = imgp.fit_circle_raw_curvature( yc, xc, xc, circ_win * PIX_PER_MM )
    
    # smooth the data
    smooth_curv_plot = imgp.smooth_data( curv_plot, smooth_win, smooth_iter )
    
    axs[1].plot( x, curv_plot, label = "Circle-Poly, no smooth" )
    axs[1].plot( x, smooth_curv_plot, label = "Circle-Poly, smooth" )
    axs[1].legend()
    axs[1].set( xlabel = 'x (mm)', ylabel = 'curvature (1/mm)' )
    
    if display:
        plt.show()
        k = input( "Does this plot look ok to proceed? (y/n) " )
        
    # if
    
    else:
        k = 'y'
        
    # else
        
    if k.lower() == 'n':
        print( "Not continuing with the curvature analysis." )
        return -1
    
    # if
    
    # Save the figure
    idx = filename.index( '.jpg' )
    outfig = filename[:idx] + '_processed.png'
    del( idx )
    fig.savefig( outfig )
    plt.close( 'all' )
    
    print( "Continuing with the curvature analysis." )
    
#     # find the active areas
#     x0 = x.max()  # start integrating from the tip
#     lb = x.min()
#     ub = x.max()
#     x_active = imgp.find_active_areas( x0, poly, active_areas, PIX_PER_MM, lb, ub )
#     x_active = np.array( x_active ).reshape( -1 )

    # find the curvature at the active areas
    curv_interp = interpolate.interp1d( x, smooth_curv_plot )
    curvature = curv_interp( x_active )
    
    # result file name processing
    outfile = '.'.join( filename.split( '.' )[:-1] ) + '.txt'
    if 'fbg' not in outfile:
        outfile = outfile.replace( 'mono', 'monofbg' )
    outfile = outfile.replace( 'mono', 'curvature_mono' )
    
    print( "Completed curvature analysis." )
    
    # write the result file
    with open( outfile, 'w+' ) as writestream:
        writestream.write( "Curvature of the needle @ the active areas.\n" )
        
        writestream.write( f"Pixels/mm: {PIX_PER_MM}\n" )
        
        writestream.write( imgpconfig + '\n' )
        
        writestream.write( "Active areas (mm): " )
        msg = np.array2string( active_areas, separator = ', ',
                                  max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
        
        writestream.write( "Curvatures (1/mm): " )
        msg = np.array2string( curvature, separator = ', ',
                                 precision = 10, max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
        
        writestream.write( "Polynomial coefficients: " )
        msg = np.array2string( poly.coef , separator = ', ',
                                 precision = 10, max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
        
        writestream.write( "x range of center line (px): " )
        msg = f"[ {x.min()}, {x.max()} ]"
        writestream.write( msg + '\n' )
        
        writestream.write( "x-values of active areas (px): " )
        msg = np.array2string( x_active, separator = ', ',
                                 precision = 10, max_line_width = np.inf ).strip( '[]' )
        writestream.write( msg + '\n' )
    
    # with
    
    print( "Wrote outfile:", outfile )
    return 0
    
# get_curvature_image

                
def get_FBGdata_windows( lutimes: np.ndarray, dt: float, timestamps: np.ndarray, max_match: int = 100 ):
    """ Function to find the windowed timedata from the gathered FBGdata.
    
        @param lutime: list of items to look up
        
        @param dt:     float, the time window length in seconds.
        
        @param timestamps: numpy 1-D array of timestamps
        
        @param max_match: (optional, default = 100) number of maximum matches
                
        @return 1-D array -> indices matching to the timestamps matched
                
    """

    retval = -1 * np.ones( ( len( lutimes ), max_match ) )  # instantiate no matches
    
    dT = timedelta( seconds = dt )
    
    for kk, lut in enumerate( lutimes ):
        idxs = np.logical_and( timestamps >= lut - dT, timestamps <= lut + dT )
        idxs = np.argwhere( idxs ).reshape( -1 )
        retval[kk, :len( idxs )] = idxs[:max_match]
        
    # for
    
    return retval

# get_FBGdata_window

    
def read_fbgData( filename: str, fbg_needle: FBGNeedle, lines: list = [-1] ):
    """ Function to read in the FBG data
    
        @param filename: str, the input fbg data file
        
        @param fbg_needle: FBGNeedle, the FBGNeedle class object for the needle
        
        @param lines: list of line numbers that would like to be read in. 
                        (default = [-1], indicating all lines)
        
        @return: (timestamps, fbgdata)
                 timestamps: numpy array of timestamps corresponding row-wise
                                to fbgdata
                 fbgdata:    numpy array of fbg readings where the rows are 
                                 the time per entry
                                 
    """
    global TIME_FMT, TIME_FIX
    
    max_lines = max( lines )
    timestamps = np.empty( 0 )
#     fbgdata = np.empty( ( 0, 3 * num_active_areas ) )
    fbgdata = np.empty( ( 0, fbg_needle.num_channels * fbg_needle.num_aa ) )  # more flexible
    
    with open( filename, 'r' ) as file:
        for i, line in enumerate( file ):
            
            # read in desired lines
            if lines != [-1]:
                
                if i > max_lines:  # passed the maximum lines, stop reading
                    break
                
                elif i not in lines:  # not a line we want to read
                    continue
            
            # if
            
            ts, datastring = line.split( ':' )
            ts = datetime.strptime( ts, TIME_FMT ) + TIME_FIX
            t0 = datetime( ts.year, ts.month, ts.day )
            dt = ( ts - t0 ).total_seconds()
            timestamps = np.append( timestamps, dt )
            data = np.fromstring( datastring, sep = ',' , dtype = float )
            
#             if len( data ) == 3 * num_active_areas:  # all active areas measured
#                 fbgdata = np.vstack( ( fbgdata, data ) )
                
            # more flexible way
            if len( data ) == fbg_needle.num_channels * fbg_needle.num_aa:  # all active areas measured
                fbgdata = np.vstack( ( fbgdata, data ) )
                
            else:  # too many or not enough sensor readings
                fbgdata = np.vstack( ( fbgdata, -1 * fbgdata.shape[1] ) )
                
        # for
    # with
    
    # remove "-1" rows - more peaks than intended
    timestamps = timestamps[fbgdata[:, 0] > 0]
    fbgdata = fbgdata[[fbgdata[:, 0] > 0]]
    
    # sort by the timestamps
    args = np.argsort( timestamps, axis = 0 )
    timestamps = timestamps[ args ]
    fbgdata = fbgdata[ args ]
    
    return timestamps, fbgdata
            
# read_fbgData


def read_needleparam( filename: str ):
    """ Function to read the needle parameters file
    
        @param filename: str, representing the needle parameter filename
        
        
        @return (needle_length, # active_areas, active_areas)
                    needle_length  = the length of the needle (in mm)
                    # active_areas = number of active areas
                    active_areas   = numpy 1-d array of the distances of active
                                         areas from the tip (in mm)
                                    
    """
    with open( filename, 'r' ) as file:
        lines = file.read().split( '\n' )
        _, needle_length, num_active_areas = lines[0].split( ',' )
        
        needle_length = float( needle_length )
        num_active_areas = int( num_active_areas )
        
        active_areas = np.fromstring( lines[1], sep = ',', dtype = float )
        
    # with
    
    return needle_length, num_active_areas, active_areas

# read_needleparam


def plot_results_stepped( shift_data: pd.DataFrame, fbg_needle: FBGNeedle,
                          title: str = "", outfile_base: str = None, show_plots: bool = False ):
    """ This is a function to plot the results for stepped insertions
    
        @param summ_data: the summary results of stepped insertion (curvature | insert. length)
        
        @param all_data: the entire data
        
        @param fbg_needle, FBGNeedle: the FBG needle parameter object
        
    """
    # get the indices
    curvature = sorted( np.array( shift_data.index.unique( level = 0 ) ) )
    step_trials = sorted( np.array( shift_data.index.unique( level = 1 ) ) )
    ch_list = list( shift_data.columns.unique( level = 0 ) )
    aa_list = list( shift_data.columns.unique( level = 1 ) )
    
    # rename indices for insertion length
    aa_active_list = ['AA1'] + ['AA1-' + aa for aa in aa_list[1:]] + ['Full Insertion']
    rn_inslength = {ins_len: aa_active for ins_len, aa_active in zip( step_trials, aa_active_list )}
    
    # reorganize per AA and prepare for plotting
    plot_shift_data = shift_data.stack( 1 ).swaplevel( 0, 2, 0 ).swaplevel( 1, 2, 0 ).sort_index( 0 )
    plot_shift_data = plot_shift_data.rename( rn_inslength, axis = 'index', level = 2 )
    plot_shift_data = plot_shift_data.loc[pd.IndexSlice[:, 0.25:, :], :].unstack( 2 ).swaplevel( 0, 1, 1 )
        
    # separate into AVG and STD
    plot_mean_shift_data = plot_shift_data.loc[:, 'Average (nm)']
    plot_std_shift_data = plot_shift_data.loc[:, 'STD (nm)']
    # plotting font size
    plt.rcParams.update( {'font.size': 16} )
    
    # plot for each aa
    for aa in aa_list:
        # subplot figure and axes over channels
        fig_aa, axs = plt.subplots( len( ch_list ), 1, sharex = True, figsize = ( 10.0, 10.0 ) )
        fig_aa.suptitle( title + " | " + aa )  
        
        # grab the aa data
        plot_mean_shift_data_aa = plot_mean_shift_data.loc[aa, :]
        plot_std_shift_data_aa = plot_std_shift_data.loc[aa, :]
        
        # plot the aa data in bars over each channel for the subplots
        for ch_i, ax_i in zip( ch_list, axs ):
            # data per channel
            plot_mean_shift_data_aa_ch = plot_mean_shift_data_aa.loc[:, ch_i]
            plot_std_shift_data_aa_ch = plot_std_shift_data_aa.loc[:, ch_i]
            
            # plot the channel data
            plot_mean_shift_data_aa_ch.plot( kind = 'bar', ax = ax_i, legend = False,
                                             yerr = plot_std_shift_data_aa_ch, capsize = 4 )
            ax_i.set_xlabel( 'Curvature (1/m)' )
            ax_i.set_ylabel( 'Signal Response (nm)' )
            ax_i.set_title( ch_i )
            
        # for
        
        handles, labels = ax_i.get_legend_handles_labels()
        fig_aa.legend( handles, labels, loc = 'upper left' )    
        
        if outfile_base is not None:
            plt.savefig( outfile_base + "_" + aa + ".png", fig = fig_aa )
            plt.close( 'all' )
            print( "Saved figure:", outfile_base + "_" + aa + ".png" )
            
        # if
            
    # for
    
    # show all the plots if we aren't saving
    if show_plots:
        plt.show()
        
    # if
    
# plot_results_stepped


def process_curvature_directory( directory: str, filefmt: str = "curvature_monofbg*.txt" ):
    """ Function to parse the curvature directory as to an Excel file."""
    col_letts = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # get files and output file
    outfile = directory + 'curvature_data.xlsx'
#     curvfiles = glob.glob( directory + filefmt ) # not needed, implmented in function already
    
    # get the Excel workbook open
    workbook = xlsxwriter.Workbook( outfile )
    result = workbook.add_worksheet( "Summary" )
    
    # get the curvature data
    curv_data = load_curvature( directory )
    
    if 0 in curv_data.shape:
        return -1  # no data
    
    # if
    
    # write the header to the file
    header = ["time (s)", "AA1 Curvature (1/mm)", "AA2 Curvature (1/mm)",
              "AA3 Curvature (1/mm)"]
    result.write_row( 0, 0, header )
    row_start_idx = 1
    
    # write the curvature data to the file
    for idx, row in enumerate( curv_data ):
        result.write_row( idx + row_start_idx, 0, row )
        
    # for
    
    # close and save the excel workbook
    workbook.close()
    print( f"Summary file written: '{outfile}'" )

    return 0
# process_curvature_directory


def process_fbgdata_directory( directory: str, fbg_needle: FBGNeedle, filefmt: str = "fbgdata*.txt", ):
    """ Process fbgdata text files """
#     time_fmt = 'fbgdata_%h'
    col_letts = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    outfile = directory + 'fbgdata.xlsx'
    fbgfiles = glob.glob( directory + filefmt )
    
    workbook = xlsxwriter.Workbook( outfile )
    result = workbook.add_worksheet( "Summary" )
    
#     header = ["time (s)",
#               "CH1 | AA1", "CH1 | AA2", "CH1 | AA3",
#               "CH2 | AA1", "CH2 | AA2", "CH2 | AA3",
#               "CH3 | AA1", "CH3 | AA2", "CH3 | AA3" ]
    
    # more direct way of performing header operation
    header = ["time(s)"]
    header.extend( ["CH{:d} | AA{:d}".format( nc + 1, na + 1 ) for ( nc, na ) in
                      itertools.product( range( fbg_needle.num_channels ), range( fbg_needle.num_aa ) )] )
    
    # excel formulas
    mean_form = '=AVERAGEIF({0}1:{0}{1},">0")'
    std_form = "=_xlfn.STDEV.S({0}1:{0}{1})"
    min_form = '=MIN({0}1:{0}{1})'
    max_form = '=MAX({0}1:{0}{1})'
    vlookup_form = "=VLOOKUP(\"{form}\",'{sheet}'!A1:{lcol}{lrow},{idx},FALSE)"
    
    vlookup_data = []
    for file in fbgfiles:
        vl_d = {}
        
        # file formatting
        file = file.replace( '\\', '/' )
        vl_d['sheet'] = file[:-4].split( '/' )[-1]
        
        # header processing
        worksheet = workbook.add_worksheet( vl_d['sheet'] )
        worksheet.write_row( 0, 0, header )
#         ts, fbgdata = read_fbgData( file, 3 )
        ts, fbgdata = read_fbgData( file, fbg_needle )  # more flexible way
        
        vl_d['time'] = ts[0]  # in seconds
        
        col_head = np.append( ts, ['Average', 'StdDev', 'Min', 'Max'] )
        worksheet.write_column( 1, 0, col_head )
        rowidx_start = 1
        
        Nrows, Ncols = fbgdata.shape
        
        # write the data matrix
        for row_idx, data in enumerate( fbgdata, rowidx_start ):
            worksheet.write_row( row_idx, 1, data )
    
        # for

#         # write the data matrix
#         for row_idx, data in enumerate( fbgdata ):
#             worksheet.write_row( rowidx_start + row_idx, 1, data )
#     
#         # for
        
        # formulas to write
        mean_formula = [mean_form.format( c, Nrows + 1 ) for c in col_letts[1:Ncols + 1]]
        std_formula = [std_form.format( c, Nrows + 1 ) for c in col_letts[1:Ncols + 1]]
        min_formula = [min_form.format( c, Nrows + 1 ) for c in col_letts[1:Ncols + 1]]
        max_formula = [max_form.format( c, Nrows + 1 ) for c in col_letts[1:Ncols + 1]]
        
        # write the formulas
        for form_idx, form_row in enumerate( [mean_formula, std_formula, min_formula, max_formula] ):
            worksheet.write_row( rowidx_start + row_idx + 1 + form_idx , 1, form_row )
            
        # for
        
        vl_d['lrow'] = rowidx_start + row_idx + 1 + form_idx + 1
        vl_d['lcol'] = col_letts[Ncols + 1]
        
        vlookup_data.append( vl_d )
    # for
    
    # process the results worksheet
    i = 2
    result_header1 = header.copy()
    while i < len( result_header1 ):
        result_header1.insert( i, '' )
        i += 2
        
    # while
    
#     result_header2 = 9 * ['Average (nm)', 'STD (nm)']
    result_header2 = fbg_needle.num_channels * fbg_needle.num_aa * ['Average (nm)', 'STD (nm)']  # more robust way
    result.write_row( 0, 0, result_header1 )
    result.write_row( 1, 1, result_header2 )
    rowstart_idx = 2
    
    for vl_idx, vl in enumerate( vlookup_data ):
        result.write( rowstart_idx + vl_idx, 0, vl['time'] )
        
        # get the average and std rows
        write_avg = [vlookup_form.format( idx = jj, form = 'Average', **vl ) for jj in range( 2, 11 )]
        write_std = [vlookup_form.format( idx = jj, form = 'StdDev', **vl ) for jj in range( 2, 11 )]
         
        # co-mingle the values
        write_val = write_std.copy()
        for idx, avg in enumerate( write_avg ):
            write_val.insert( 2 * idx, avg )
            
        # for
        
        result.write_row( rowstart_idx + vl_idx, 1, write_val )
            
    # for
    workbook.close()
    print( 'Wrote fbg data file:', outfile )
    
# process_fbgdata_directory


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
    avg_mask = proc_data.columns.get_level_values( 2 ) == data_head[0]  # for col selection
    std_mask = proc_data.columns.get_level_values( 2 ) == data_head[1]  # for col selection
    
    # process each frame of data from the trials to create a summary table
    for i, sheet in enumerate( data_sheets ):
        sheet_data = wb[sheet]
        sheet_data = sheet_data[sheet_data.iloc[:, 0].str.isalpha() == False].astype( float )  # and convert all to floats
        
        # change the headers for better alignment
        sheet_data.columns = proc_header.droplevel( 2 ).drop_duplicates() 
        
        # add the data to the processed data
        proc_data.iloc[i, 0] = sheet_data.iloc[:, 0].min()  # set the time
        proc_data.iloc[i, avg_mask] = sheet_data.iloc[:, 1:].mean()  # set the mean
        proc_data.iloc[i, std_mask] = sheet_data.iloc[:, 1:].std()  # set the STD  
    
    # for
    
    return proc_data

# process_fbgdata_file


def process_fbgdata_file_stepped( fbg_input_file: str, step_trials: list, ref_trials: int, fbg_needle: FBGNeedle ):
    """ 
    This function is to handle the fbgdata.xlsx files and combine them into a single file.
    
    @param fbg_input_file, str: The fbgdata file to be processed
    
    @param step_trials, list: list of insertion length for stepped insertion
    
    @param ref_trials, the reference number of trials (0 curvature)
    
    @param fbg_needle, FBGNeedle: The FBGNeedle param class object
    
    @return: The Pandas Dataframe of the Avg. and STD for the fbg_input_file 
    """
    wb = pd.read_excel( fbg_input_file, None )  # read all sheets from wkbook
    data_sheets = [k for k in wb.keys() if k != 'Summary']  # only count the data sheets
    
    # Note: remove the end parts of the files (remove the bottom 4 when processing Avg. Std. Min. Max.)
    
    # initialize the DataFrame for all the processed data
    first_head = ['Insertion Length (mm)', 'time (s)']
    time_head = pd.MultiIndex.from_arrays( [first_head, len( first_head ) * ['empty 2'],
                                            len( first_head ) * ['empty 3']] )
    ch_head = ['CH' + str( i + 1 ) for i in range( fbg_needle.num_channels )]
    aa_head = ['AA' + str( i + 1 ) for i in range( fbg_needle.num_aa )]
    data_head = ['Average (nm)', 'STD (nm)']
    proc_header = pd.MultiIndex.from_product( [ch_head, aa_head, data_head] )
    read_header = time_head[0:1].append( proc_header )
    proc_header = time_head.append( proc_header )
    
    # the empty processed data
    proc_data = pd.DataFrame( index = range( len( data_sheets ) ), columns = proc_header )
    avg_mask = proc_data.columns.get_level_values( 2 ) == data_head[0]  # for col selection
    std_mask = proc_data.columns.get_level_values( 2 ) == data_head[1]  # for col selection
    
    # process each frame of data from the trials to create a summary table
    if len( data_sheets ) == ref_trials:  # reference full insertion trial
        for i, sheet in enumerate( data_sheets ):
            sheet_data = wb[sheet]
            sheet_data = sheet_data[sheet_data.iloc[:, 0].str.isalpha() == False].astype( float )  # and convert all to floats
            
            # change the headers for better alignment
            sheet_data.columns = read_header.droplevel( 2 ).drop_duplicates() 
            
            # add the data to the processed data
            proc_data.iloc[i, 0] = fbg_needle.length  # set the insertion length
            proc_data.iloc[i, 1] = sheet_data.iloc[:, 0].min()  # set the time
            proc_data.iloc[i, avg_mask] = sheet_data.iloc[:, 1:].mean()  # set the mean
            proc_data.iloc[i, std_mask] = sheet_data.iloc[:, 1:].std()  # set the STD  
        
        # for
    # if
    
    elif len( data_sheets ) == len( step_trials ):  # non-zero curvature w/ correct num of trials
        # the columns for each step
        for i, sheet in enumerate( data_sheets ):
            sheet_data = wb[sheet]
            sheet_data = sheet_data[sheet_data.iloc[:, 0].str.isalpha() == False].astype( float )  # and convert all to floats
            
            # change the headers for better alignment
            sheet_data.columns = read_header.droplevel( 2 ).drop_duplicates() 
            
            # add the data to the processed data
            proc_data.iloc[i, 0] = step_trials[i]  # set the insertion length
            proc_data.iloc[i, 1] = sheet_data.iloc[:, 0].min()  # set the time
            proc_data.iloc[i, avg_mask] = sheet_data.iloc[:, 1:].mean()  # set the mean
            proc_data.iloc[i, std_mask] = sheet_data.iloc[:, 1:].std()  # set the STD  
            
        # for
    # elif
    
    else:
        raise IndexError( "The number of data frames don't match the expected amount." )  
    
    # else
          
    return proc_data

# process_fbgdata_file


def main_camera():
    skip_prev = True
    show_imgp = True
    correct_firstlast = False

    directory = "../FBG_Needle_Calibration_Data/needle_1/"
    
    needleparam = directory + "needle_params.csv"
    num_actives, length, active_areas = read_needleparam( needleparam )
    
    fbg_needle = FBGNeedle.load_json( directory + "needle_params.json" )  # load the FBGNeedle file
    
    directory += "Validation/Sanity_Check/"
    directory += "01-03-20_11-23/"
    
    imgfiles = glob.glob( directory + "monofbg*.jpg" )
    imgfiles.sort()
#     imgfiles = [directory + "mono_0006.jpg"]  # for testing
    
    img_patt = r"monofbg_([0-9][0-9])-([0-9][0-9])-([0-9]+)_([0-9][0-9])-([0-9][0-9])-([0-9][0-9]).([0-9]+).jpg"
    retval = 0
    for idx, imgf in enumerate( imgfiles ):
        try:
            mon, day, yr, hr, mn, sec, ns = re.search( img_patt, imgf ).groups()
            curv_file = directory + f"curvature_monofbg_{mon}-{day}-{yr}_{hr}-{mn}-{sec}.{ns}.txt"
        
        except:
            curv_file = ''
            pass
        
        if not ( skip_prev  and os.path.exists( curv_file ) ):
            print( "Processing file:" , imgf )
#             str_ts = f"{hr}:{mn}:{sec}.{ns[0:6]}"
#             ts = datetime.strptime( str_ts, "%H:%M:%S.%f" )
#             print( ts )
            if idx == 0 or idx == len( imgfiles ) - 1:
                retval += get_curvature_image( imgf, np.array( fbg_needle.sensor_location ), fbg_needle.length, show_imgp, polfit = 3 )
            
            else:
                retval += get_curvature_image( imgf, np.array( fbg_needle.sensor_location ), fbg_needle.length, show_imgp, polfit = 4 )
            
            print()
        # if
        
#         else:
#             print( f"Curvature file exists '{curv_file}'.\n" )
    # for
    
    if retval >= 0:
        process_curvature_directory( directory )
        process_fbgdata_directory( directory )
        
    # if
        
# main_camera


def main_jig_stepped( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvature_values: list,
              ref_trials: int, step_trials: list, num_trials: int, savefile_base: str, Tcorr: bool = True ):
    """ Progressive jig calibration/validation """
    # SET-UP
        
    # correct the formatting of the directories
    for exp_angle, dirs in dirs_degs.items():
        dirs_degs[exp_angle] = [d.replace( '\\', '/' ) + '/' if not d.endswith( '/' ) else d.replace( '\\', '/' ) for d in dirs]
        
    # for
        
    # iterate through the directories processing the fbg data files individually
    for d in sum( dirs_degs.values(), [] ):
        if os.path.isdir( d ):
            print( 'Processing:', d )
            process_fbgdata_directory( d, fbg_needle )
            print()
            
        # if
    # for
    
    # consolidate the fbgdata_files
    for exp_angle, fbgdata_dir in dirs_degs.items():
        print( "Handling angle:", exp_angle, "degs" )
        fbgdata_files = [d + "fbgdata.xlsx" for d in fbgdata_dir]
        outfile_base = directory + savefile_base + "_{0:d}deg".format( exp_angle )
        out_fbgresult_file = outfile_base + '.xlsx'
        
        # consolidate the stepped fbgdata files
        summ_data, *_ = consolidate_fbgdata_files_stepped( fbgdata_files, curvature_values * num_trials,
                                                                    step_trials, ref_trials,
                                                                    fbg_needle, out_fbgresult_file ) 
        print( "Saved:", out_fbgresult_file )
        
        # calculate the wavelength shifts
        shift_data = calculate_shift_stepped( summ_data, fbg_needle, Tcorr = Tcorr )
        
        # plot and save the results
        plt_title = "T Corr Signal Response" if Tcorr else "Signal Response"
        plot_results_stepped( shift_data, fbg_needle, plt_title, outfile_base, False )
        
        print()
    
    # for
    
# main_jig_stepped


def main_jig( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvature_values: list,
              savefile_base: str ):
    """ Non-progressive jig calibration/validation """
    # SET-UP
    
    # correct the formatting of the directories
    for exp_angle, dirs in dirs_degs.items():
        dirs_degs[exp_angle] = [d.replace( '\\', '/' ) + '/' if not d.endswith( '/' ) else d.replace( '\\', '/' ) for d in dirs]
        
    # for
    
    # iterate through the directories processing the fbg data files individually
    for d in sum( dirs_degs.values(), [] ):
        if os.path.isdir( d ):
            print( 'Processing:', d )
            process_fbgdata_directory( d, fbg_needle )
            print()
            
        # if
    # for
    
    for exp_angle, fbgdata_dir in dirs_degs.items():
        print( "Handling angle:", exp_angle, "degs" )
        
        # set up the fbg data files and output file
        fbgdata_files = [d + "fbgdata.xlsx" for d in fbgdata_dir]
        outfile_base = directory + savefile_base + "_{0:d}deg".format( exp_angle )
        out_fbgresult_file = outfile_base + '.xlsx'
        
        # consolidate the fbgdata files
        consolidate_fbgdata_files( fbgdata_files, curvature_values, fbg_needle,
                              out_fbgresult_file )
        print( "Saved:", out_fbgresult_file )
        print()
        
    # for
    
# main_jig


if __name__ == '__main__':    
    # SET-UP 
    directory = "../FBG_Needle_Calibration_Data/needle_3CH_4AA/"
    fbg_needle = FBGNeedle.load_json( directory + "needle_params.json" )  # load the fbg needle json
    
    print( "FBG Needle Parmeters:" )
    print( fbg_needle )
    print( 75 * '=' )
    print()
    
    pass
    
    # curvatures for each jig
    curvature_values = {'cal': [0, 0.5, 1.6, 2.0, 2.5, 3.2, 4],
                        'val': [0, 0.25, 0.8, 1.0, 1.25, 3.125]}
    
    # progressive insertion: AA1, AA1-2, AA1-3, AA1-4, Full Insertion dict (length: # trials)
    prog_insertion = True
    num_trials = 10
    ref_trials = 5
    step_trial_dict = {15: 5, 48: 5, 83: 5, 110: 5, fbg_needle.length: 5}
    step_trials = [[v] * k for v, k in step_trial_dict.items()]  # list-ify the dict
    step_trials = sum( step_trials, [] )  # flatten
    
    # FBG data experiment directory for processing
#     directory += "Validation_Jig_Progressive_Insertion_08-26_29-20/"
    directory += "Jig_Calibration_08-05-20/"
    
    # gather the directories contatining the fbgdata .txt files
    dirs_degs = {}
    dirs_degs[0] = glob.glob( directory + "0_deg/Trial_*/08*" )
    dirs_degs[90] = glob.glob( directory + "90_deg/Trial_*/08*" )
    dirs_degs[180] = glob.glob( directory + "180_deg/Trial_*/08*" )
    dirs_degs[270] = glob.glob( directory + "270_deg/Trial_*/08*" )
    
    # correct the formatting of the directories \ (or '\\') -> /
    for exp_angle, dirs in dirs_degs.items():
        dirs_degs[exp_angle] = [d.replace( '\\', '/' ) + '/' if not d.endswith( '/' ) else d.replace( '\\', '/' ) for d in dirs]
        
    # for
    
    # main method 
    if prog_insertion:
        main_jig_stepped( directory, dirs_degs, fbg_needle, curvature_values['cal'], ref_trials, step_trials, num_trials,
                     savefile_base = "08-26_29-20_Prog_Insertion_FBGResults", Tcorr = True )
    
    # if    
        
    else:
        main_jig( directory, dirs_degs, fbg_needle, curvature_values['val'],
                  savefile_base = "08-26_29-20_Prog_Insertion_FBGResults" )
        
    # else
    
    print( "Program has terminated." )  
    
    #===========================================================================
    # # iterate through the directories processing the fbg data files individually
    # for d in []:  # sum( dirs_degs.values(), [] ):
    #     if os.path.isdir( d ):
    #         print( 'Processing:', d )
    #         process_fbgdata_directory( d, fbg_needle )
    #         print()
    #         
    #     # if
    # # for
    # 
    # # consolidate the fbgdata_files
    # for exp_angle, fbgdata_dir in []:  # dirs_degs.items():
    #     print( "Handling angle:", exp_angle, "degs" )
    #     fbgdata_files = [d + "fbgdata.xlsx" for d in fbgdata_dir]
    #     outfile_base = directory + "08-26_29-20_Prog_Insertion_FBGResults_{0:d}deg".format( exp_angle )
    #     out_fbgresult_file = outfile_base + '.xlsx'
    #     
    #     if prog_insertion:  # progressive insertion
    #         summ_data, _, all_data = consolidate_fbgdata_files_stepped( fbgdata_files, curvature_values['val'] * 10,
    #                                                                     step_trials, ref_trials,
    #                                                                     fbg_needle, out_fbgresult_file ) 
    #         shift_data = calculate_shift_stepped( summ_data, fbg_needle, Tcorr = True )
    #         plot_results_stepped( shift_data, fbg_needle, "T Corr. Signal Response", outfile_base, False )
    #         
    #     # if
    #     
    #     else:  # non-progressive insertion
    #         consolidate_fbgdata_files( fbgdata_files, curvature_values['val'], fbg_needle,
    #                           out_fbgresult_file ) 
    #         
    #     # else
    #     
    #     print( "Saved:", out_fbgresult_file )
    #     print()
    # 
    # # for
    #===========================================================================
    
# if
    

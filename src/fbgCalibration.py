'''
Created on Dec 6, 2019

@author: Dimitri Lezcano

@summary: This script is intended for the use of performing the data processing
          incorporating the image and FBG data when needle calibration is 
          performed.
'''
import glob, re, cv2, xlsxwriter, os.path
import numpy as np
import matplotlib.pyplot as plt
import calibrationMatrix as calmat
import image_processing as imgp
from datetime import datetime, timedelta
from scipy import interpolate
from FBGNeedle import FBGNeedle
import itertools
# from needle_segmentation_script import CROP_AREA

TIME_FMT = "%H-%M-%S.%f"
TIME_FIX = timedelta( hours = 0 )  # the internal fix for the time

# PIX_PER_MM = 8.875 # what appears to be the actual fit
PIX_PER_MM = 8.498439767625596
# CROP_AREA = ( 32, 425, 1180, 600 )
CROP_AREA = ( 40, 420, 1230, 620 )
BO_REGIONS = [( 0, 0, -1, 30 ), ( 1060, 125, -1, -1 )]
BO_REGIONS.append( ( 1090, 105, -1, -1 ) )
BO_REGIONS.append( ( 0, 75, 145, -1 ) )
BO_REGIONS.append( ( 950, 0, -1, 68 ) )
BO_REGIONS.append( ( 0, 170, -1, -1 ) )


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
    header.append( ["CH{:d} | AA{:d}".format( nc + 1, na + 1 ) for ( nc, na ) in
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
        ts, fbgdata = read_fbgData( file, fbg_needle.num_aa )  # more flexible way
        
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


def main():
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
        
# main


if __name__ == '__main__':
    # main()
    
    directory = "../FBG_Needle_Calibration_Data/needle_1/"
    fbg_needle = FBGNeedle.load_json( directory + "needle_params.json" )  # load the fbg needle json
    
    # process the FBG data directory
    directory += "Jig_Calibration/"
    # directory += "Calibration/0 deg/"
#     directory +="12-28-19_14-43/"
    directories = glob.glob( directory + '01-*' )
    for dir in directories:
        if os.path.isdir( dir ):
            dir += '/'
            print( 'Processing:', dir )
            process_fbgdata_directory( dir, fbg_needle )
            print()
            
        # if
#     # for
#     
#     process_fbgdata_directory( directory )
    
    print( "Program has terminated." )
    
# if
    

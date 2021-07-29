"""
Created on Jul 28, 2021

@author: Dimitri Lezcano

@summary: This is a script to perform FBG needle calibration

"""
import argparse
import glob
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fbg_signal_processing
import open_files
from FBGNeedle import FBGNeedle


def _linear_fit( x, y ) -> (float, float, float):
    """ Perform a linear fit and return the R^2 value

        Args:
            x: the x-coordinates
            y: the y-coordinates

        Return:
            m, b, R^2

    """
    coeffs = np.polyfit( x, y, 1, rcond=None )
    R_sq = np.corrcoef( x, y )[ 0, 1 ] ** 2  # R^2

    return coeffs[ 0 ], coeffs[ 1 ], R_sq


# _linear_fit


def combine_fbgdata_directory( directory: str, num_channels: int, num_active_areas: int, save: bool = False ):
    """ Combine the FBG data files from an entire directory """
    fbgdata_files = glob.glob( os.path.join( directory, "fbgdata*.txt" ) )

    # get all of the data frames
    summary_df = None
    fbgdata_dfs = [ ]
    for file in fbgdata_files:
        df = open_files.read_fbgdata( file, num_channels, num_active_areas )
        fbgdata_dfs.append( df )

        # create the summary df table if not already
        if summary_df is None:
            summary_df = df.copy().drop( range( df.shape[ 0 ] ), axis=0 )

        # if

        # add the mean values to the summary_df
        new_row = df.mean()  # mean peak values
        new_row[ 'time' ] = df[ 'time' ].max()  # latest time

        summary_df = summary_df.append( new_row, ignore_index=True )

    # for

    if save:
        outfile = os.path.join( directory, 'fbgdata.xlsx' )

        # save each df to an excel file
        with pd.ExcelWriter( outfile ) as xl_writer:
            # write the summary table
            summary_df.to_excel( xl_writer, sheet_name='Summary' )

            for file, df in zip( fbgdata_files, fbgdata_dfs ):
                filename = os.path.split( file )[ -1 ]
                df.to_excel( xl_writer, sheet_name=filename )

            # for
        # with
        print( f"Saved FBG data file: {outfile}" )

    # if

    return summary_df, fbgdata_dfs


# combine_fbgdata_directory

def combine_fbgdata_summary( fbg_summary_files: list, curvature_values: list, num_channels: int, num_active_areas: int,
                             outfile: str = None
                             ) -> (pd.DataFrame, pd.DataFrame):
    """ Function to combine the FBG data summary files

        Args:
            fbg_summary_files: list of the summary files to be found
            curvature_values: list of the curvature values
            num_channels: int of the number of channels for the FBG needle
            num_active_areas: int of the number of active areas for the FBG needle
            outfile: str (Default: None) the output file for the results
    """
    # configure the result data frames
    ch_aa_head, ch_head, aa_head = FBGNeedle.generate_ch_aa( num_channels, num_active_areas )
    data_head = [ 'Average (nm)', 'STD (nm)' ]  # unused, not implemented for STD

    header = [ 'Curvature (1/m)', 'time' ] + ch_aa_head

    all_data_df = pd.DataFrame( np.zeros( (0, len( header )) ), columns=header )

    for fbgdata_file, curvature in zip( fbg_summary_files, curvature_values ):
        df = pd.read_excel( fbgdata_file, sheet_name='Summary', index_col=0 )  # load in the df

        df[ header[ 0 ] ] = curvature  # add in the curvatures column

        all_data_df = all_data_df.append( df, ignore_index=True )  # append the new dataframe

    # for

    # create the summary data frame by averaging the curvatures
    summary_df = all_data_df.groupby( header[ 0 ] ).mean()
    summary_df[ 'time' ] = all_data_df.groupby( header[ 0 ] ).max()[ 'time' ]
    summary_df = summary_df.reset_index()[ header ]

    # save the output file
    if outfile is not None:
        with pd.ExcelWriter( outfile ) as xl_writer:
            summary_df.to_excel( xl_writer, sheet_name='Summary' )
            all_data_df.to_excel( xl_writer, sheet_name='Trial Data' )

        # with

        print( "Saved consolidated FBG data file:", outfile )

    # if

    return summary_df, all_data_df


# combine_fbgdata_summary

def jig_calibration( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list ) -> FBGNeedle:
    """ Perform jig calibration """
    # process the FBG signals
    total_df, _, proc_Tcomp_total_df = jig_process_signals( directory, dirs_degs, fbg_needle, curvatures_list )

    # TODO: perform calibration via least squares formulation

    # TODO: output results to a log file

    # TODO: update and save the calibrated fbg_needle

    return fbg_needle


# jig_calibration

def jig_process_signals( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list ) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """ Process the FBG signals """
    # set-up
    aa_assignments = np.array( fbg_needle.assignments_AA() )
    ch_assignments = np.array( fbg_needle.assignments_CH() )
    ch_aa_name, *_ = fbg_needle.generate_chaa()

    # process all of the FBG data files
    for d in sum( dirs_degs.values(), [ ] ):
        if os.path.isdir( d ):
            print( "Processing FBG data directory:", d )
        if os.path.exists( os.path.join( d, 'fbgdata.xlsx' ) ):
            print( "Already processed. Continuing..." )

        # if
        else:
            combine_fbgdata_directory( d, fbg_needle.num_channels, fbg_needle.num_aa, save=True )
            print( 'Completed FBG data directory:', d )

        # else
        print()

    # for

    # iterate through each of the angles to get the FBG data files to summarize the curvature trials
    total_df = None
    header = [ 'angle', 'Curvature (1/m)', 'time' ] + ch_aa_name
    for angle, fbgdata_dir in dirs_degs.items():
        fbgdata_files = [ os.path.join( d, 'fbgdata.xlsx' ) for d in fbgdata_dir ]
        outfile_base = os.path.join( directory, f"FBGResults_{angle}_deg" )
        out_fbgresult_file = outfile_base + ".xlsx"

        # consolidate the FBG data directories into one FBG file
        summary_df, *_ = combine_fbgdata_summary( fbgdata_files, curvatures_list, fbg_needle.num_channels,
                                                  fbg_needle.num_aa, out_fbgresult_file
                                                  )
        summary_df[ 'angle' ] = angle  # add the angle measurement
        summary_df = summary_df[ header ]  # reorganize the df

        # add the summary dataframe to the total trials
        if total_df is None:
            total_df = summary_df

        # if

        else:
            total_df = total_df.append( summary_df, ignore_index=True )

        # else

        # Plot and save linear fits wavelength shift vs curvature and curvature per angle
        fig, axs = plt.subplots( fbg_needle.num_aa, sharex='col' )
        fig.set_size_inches( [ 13, 8 ] )
        for aa_i in range( 1, fbg_needle.num_aa + 1 ):
            mask_signals = (aa_i == aa_assignments)  # determine which columns of the FBG signals to get
            mask = np.append( [ False, True, False ], mask_signals )  # include the curvatures and remove time

            summary_df.iloc[ :, mask ].plot( x='Curvature (1/m)', style='.', ax=axs[ aa_i - 1 ] )
            axs[ aa_i - 1 ].set_ylabel( 'signal (nm)' )

            # plot linear fits
            line_colors = { c.get_label(): c.get_color() for c in axs[ aa_i - 1 ].get_children() if
                            isinstance( c, mpl.lines.Line2D ) }
            for ch_aa_idx in np.where( np.append( 3 * [ False ], mask_signals ) )[ 0 ]:
                # get the linear fit
                m, b, R_sq = _linear_fit( summary_df[ 'Curvature (1/m)' ], summary_df.iloc[ :, ch_aa_idx ] )
                k = np.linspace( summary_df[ 'Curvature (1/m)' ].min(), summary_df[ 'Curvature (1/m)' ].max(), 100 )
                sig = m * k + b

                # get the current plot color
                ch_aa = summary_df.columns[ ch_aa_idx ]
                color = line_colors[ ch_aa ]

                # plot the linear fit
                axs[ aa_i - 1 ].plot( k, sig, color=color, label=ch_aa + " linear" )
            # for

            axs[ aa_i - 1 ].legend()  # turn on the legend (if not already)

        # for
        axs[ -1 ].set_xlabel( 'Curvature (1/m)' )
        fig.suptitle( f"{angle} deg: signals" )

        # save the figure
        outpath = os.sep.join( os.path.normpath( fbgdata_dir[ 0 ] ).split( os.sep )[ :-2 ] )
        outfile_fig = os.path.join( outpath, f"{angle}_deg_aa-signals-curvature.png" )
        fig.savefig( outfile_fig )
        print( "Saved figure:", outfile_fig )
        plt.close( fig=fig )

        print()

    # for

    # Determine the FBG Data matrices
    # determine the curvature vectors based on the angles
    header = [ 'angle', 'Curvature (1/m)', 'Curvature_x (1/m)', 'Curvature_y (1/m)', 'time' ] + ch_aa_name
    total_df[ header[ 2 ] ] = total_df[ 'Curvature (1/m)' ] * np.cos( np.deg2rad( total_df[ 'angle' ] ) ).round( 10 )
    total_df[ header[ 3 ] ] = total_df[ 'Curvature (1/m)' ] * np.sin( np.deg2rad( total_df[ 'angle' ] ) ).round( 10 )
    total_df = total_df[ header ]  # reorganize the table

    # perform signal processing
    ref_signals = total_df.loc[ total_df[ 'Curvature (1/m)' ] == 0 ]  # get the reference signals per experiment

    proc_total_df = total_df.copy()
    proc_Tcomp_total_df = total_df.copy()
    for angle in np.unique( ref_signals[ 'angle' ] ):
        # get the reference signal and raw signals
        ref_signal = ref_signals.loc[ ref_signals[ 'angle' ] == angle, ch_aa_name ].to_numpy()
        signals = total_df.loc[ total_df[ 'angle' ] == angle, ch_aa_name ].to_numpy()

        # process the signals
        signal_shifts = fbg_signal_processing.process_signals( signals, ref_signal )
        Tcomp_signal_shifts = fbg_signal_processing.temperature_compensation( signal_shifts,
                                                                              fbg_needle.num_channels,
                                                                              fbg_needle.num_aa
                                                                              )

        # change the values for the signals
        proc_total_df.loc[ proc_total_df[ 'angle' ] == angle, ch_aa_name ] = signal_shifts
        proc_Tcomp_total_df.loc[ proc_Tcomp_total_df[ 'angle' ] == angle, ch_aa_name ] = Tcomp_signal_shifts

    # for

    # save the tables
    outfile_total_df = os.path.join( directory, 'all-curvatures-signals.xlsx' )
    with pd.ExcelWriter( outfile_total_df ) as xl_writer:
        total_df.to_excel( xl_writer, sheet_name='Raw Signals' )
        proc_total_df.to_excel( xl_writer, sheet_name='Processed Signals' )
        proc_Tcomp_total_df.to_excel( xl_writer, sheet_name='T Comp. Processed Signals' )

    # with
    print( f"Saved data file: {outfile_total_df}" )

    # TODO: Plot and save linear fits wavelength shift vs curvature_x and curvature_y

    # TODO: Plot and save linear fits for T Comp. wavelength shift vs curvature_x and curvature_y

    return total_df, proc_total_df, proc_Tcomp_total_df


# jig_process_signals

def jig_validation( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list ):
    """ Perform jig validation"""
    # process the FBG signals
    total_df, _, proc_Tcomp_total_df = jig_process_signals( directory, dirs_degs, fbg_needle, curvatures_list )

    # TODO: evaluate the validation


# jig_validation

def main( args=None ):
    # check args
    if args is None:
        return

    # if

    # set-up
    dir_pattern = r".*([0-9]+)_deg{0}([0-9].?[0-9]*){0}.*".format( os.sep.replace( '\\', '\\\\' ) )

    # display the arguments
    print( 'angles:', args.angles )
    print( 'needle_json:', args.fbgParamFile )
    print( 'calibDirectory:', args.calibDirectory )
    print( 'validDirectory:', args.validDirectory )
    print( 'calibCurvatures:', args.calib_curvatures )
    print( 'validCurvatures:', args.valid_curvatures )
    print()

    # load FBGNeedle
    fbg_needle = FBGNeedle.load_json( args.fbgParamFile )
    print( "Current FBG Needle:" )
    print( fbg_needle )
    print()

    # Prepare calibration directories
    calib_dirs_degs = { }
    for angle in args.angles:
        calib_dirs_degs[ angle ] = [ ]

        # filter out the non-applicable curavtures
        for d in glob.glob( os.path.join( args.calibDirectory, f'{angle}_deg', '*/*' ) ):
            res = re.search( dir_pattern, os.path.normpath( d ) )  # search for the path pattern
            if res is not None:
                ang, curv = res.groups()
                ang, curv = float( ang ), float( curv )  # convert strings to floats

                # if this is not a valid calibration curvature
                if curv in args.calib_curvatures:
                    calib_dirs_degs[ angle ].append( os.path.normpath( d ) )
                # if
            # if
        # for
        print( angle, ':', calib_dirs_degs[ angle ] )
        print()
    # for

    # Perform the jig calibration
    fbg_needle = jig_calibration( args.calibDirectory, calib_dirs_degs, fbg_needle, args.calib_curvatures )

    # check if validation is configured, if so, perform validation
    if (len( args.valid_curvatures ) > 0) and (args.validDirectory is not None):
        # Prepare validation directories
        valid_dirs_degs = { }

        for angle in args.angles:
            valid_dirs_degs[ angle ] = [ ]

            # filter out the non-applicable curavtures
            for d in glob.glob( os.path.join( args.validDirectory, f'{angle}_deg', '*/*' ) ):
                res = re.search( dir_pattern, os.path.normpath( d ) )  # search for the path pattern
                if res is not None:
                    ang, curv = res.groups()
                    ang, curv = float( ang ), float( curv )  # convert strings to floats

                    # if this is not a valid calibration curvature
                    if curv in args.calib_curvatures:
                        valid_dirs_degs[ angle ].append( os.path.normpath( d ) )
                    # if
                # if
            # for
        # for

        # TODO: ensure that the FBG needle is up-to-date with the calibration matrices

        # TODO: perform jig validation
        jig_validation( args.validDirectory, valid_dirs_degs, fbg_needle, args.valid_curvatures )

    # if


# main

if __name__ == "__main__":
    # Setup parsed arguments
    parser = argparse.ArgumentParser( description="Perform needle jig calibration" )
    parser.add_argument( 'fbgParamFile', type=str, help='FBG parameter json file' )

    parser.add_argument( '--valid-directory', dest='validDirectory', type=str, default=None,
                         help='Directory where all of the validation data is'
                         )
    parser.add_argument( 'calibDirectory', type=str, help='Directory where all of the calibration data is.' )

    parser.add_argument( '--angles', type=int, nargs='+', default=[ 0, 90, 180, 270 ],
                         help='Angles that will be used for the calibration'
                         )
    parser.add_argument( '--calib-curvatures', type=float, nargs='+', default=[ 0, 0.5, 1.6, 2.0, 2.5, 3.2, 4 ],
                         help='curvatures of the calibration to be used'
                         )
    parser.add_argument( '--valid-curvatures', type=float, nargs='+', default=[ 0, 0.25, 0.8, 1.0, 1.25, 3.125 ],
                         help='curvatures of the validation to be used'
                         )

    # parse the arguments
    args = parser.parse_args()

    # call main method
    main( args=args )

# if __main__

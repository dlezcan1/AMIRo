"""
Created on Jul 28, 2021

@author: Dimitri Lezcano

@summary: This is a script to perform FBG needle calibration

"""
import numpy as np
import pandas as pd

import glob
from os import path

import open_files
from FBGNeedle import FBGNeedle


def combine_fbgdata_directory( directory: str, num_channels: int, num_active_areas: int, save: bool = False ):
    """ Combine the FBG data files from an entire directory """
    fbgdata_files = glob.glob( path.join( directory, "*.txt" ) )

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
        outfile = path.join( directory, 'fbgdata.xlsx' )

        # save each df to an excel file
        with pd.ExcelWriter( outfile ) as xl_writer:
            # write the summary table
            summary_df.to_excel( xl_writer, sheet_name='Summary' )

            for file, df in zip( fbgdata_files, fbgdata_dfs ):
                filename = path.split( file )[ -1 ]
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
    ch_aa_head, ch_head, aa_head = open_files.generate_ch_aa( num_channels, num_active_areas )
    data_head = [ 'Average (nm)', 'STD (nm)' ]  # unused, not implemented for STD

    header = [ 'Curvature (1/m)', 'time' ] + ch_aa_head

    all_data_df = pd.DataFrame( columns=header )

    for fbgdata_file, curvature in zip( fbg_summary_files, curvature_values ):
        df = pd.read_excel( fbgdata_file, sheet_name='Summary' )  # load in the df

        df[ header[ 0 ] ] = curvature * np.ones( df.shape[ 0 ] )  # add in the curvatures column

        all_data_df = all_data_df.append( df, ignore_index=True )  # append the new dataframe

    # fbgdata_file

    # create the summary data frame by averaging the curvatures
    summary_df = all_data_df.groupby( header[ 0 ] ).mean()
    summary_df[ 'time' ] = all_data_df.groupby( header[ 0 ] ).max()[ 'time' ]
    summary_df = summary_df[ header ]

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

def jig_calibration( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list, savefile_base: str
                     ):
    # process all of the FBG data files
    for d in sum( dirs_degs.values(), [ ] ):
        if path.isdir( d ):
            print( "Processing FBG data directory:", d )
            combine_fbgdata_directory( d, fbg_needle.num_channels, fbg_needle.num_aa, save=True )
            print( 'Completed FBG data directory:', d )
            print()

        # if
    # for

    # iterate through each of the angles to get the FBG data files to summarize the curvature trials
    total_df = None
    for angle, fbgdata_dir in dirs_degs.items():
        fbgdata_files = [ path.join( d, 'fbgdata.xlsx' ) for d in fbgdata_dir ]
        outfile_base = path.join( directory, savefile_base + f"{angle}_deg" )
        out_fbgresult_file = outfile_base + ".xlsx"

        # consolidate the FBG data directories into one FBG file
        summary_df, *_ = combine_fbgdata_summary( fbgdata_files, curvatures_list, fbg_needle.num_channels,
                                                  fbg_needle.num_aa, out_fbgresult_file
                                                  )

        # add the summary dataframe to the total trials
        if total_df is None:
            total_df = summary_df

        # if

        else:
            total_df = total_df.append( summary_df, ignore_index=True )

        # else

        # TODO: Plot and save linear fits wavelength shift vs curvature and curvature per angle
        # TODO: plots per channel

        # TODO: plots per active area


        print()

    # for



    # TODO: write out the FBG data matrices file

    # TODO: Plot and save linear fits wavelength shift vs curvature_x and curvature_y

    # TODO: perform calibration via least squares formulation

    # TODO: perform validation


# jig_calibration

def main( args=None ):
    # TODO: argument parsing

    # TODO: load FBGNeedle

    # TODO: jig calibration
    pass


# main

if __name__ == "__main__":

    pass

# if __main__

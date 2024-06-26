"""
Created on Jul 28, 2021

Author: Dimitri Lezcano

Summary: This is a script to perform FBG needle calibration along with a class implementation of
         the Jig Calibration procedure

"""
import argparse
import glob
import itertools
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import fbg_signal_processing
import open_files
from needle_shape_sensing.sensorized_needles import (
    FBGNeedle,
    MCFNeedle,
)


class FBGNeedleJigCalibrator:
    """
        fbg_needle: FBGNeedle to be calibratied

        calib_angles: the calibration jig angles used
        valid_angles: the validation angles used

        calib_directory: the calibration jig directory
        valid_directory: the validation jig directory

        calib_curvatures: the curvatures that will be used for needle calibration
        valid_curvatures: the curvatures that will be used to validate the needle calibration

        weight_rule: the curvature weighting rule for the jig calibration
        weighted:    whether curvature weighting is configured

        DataFrames:
            total_df:            the raw signal means of the individual insertion trials
            proc_total_df:       the processed signal means of the individual insertion trials
            proc_Tcomp_total_df: the temperature compensated processed signal means of the individual insertion trials
            calval_df:           The dataframe used for calibration and validation.

    """
    directory_pattern = r"([0-9]+.?[0-9]*)_deg{0}([0-9]+.?[0-9]*)".format(
            os.sep.replace( '\\', '\\\\' )
    )  # directory pattern file structure

    def __init__( self, fbg_needle: FBGNeedle,
                  calib_directory: str, calib_curvatures: list, calib_angles: list, weight_rule: callable = None,
                  valid_directory: str = None, valid_curvatures: list = None, valid_angles: list = None ):
        self.fbg_needle = fbg_needle
        self.calib_angles = calib_angles  # experimental angles
        self.calib_directory = os.path.normpath( calib_directory )
        self.calib_curvatures = calib_curvatures
        self.valid_directory = os.path.normpath( valid_directory ) if valid_directory is not None else valid_directory
        self.valid_curvatures = valid_curvatures
        self.valid_angles = valid_angles

        # weighting rule
        if weight_rule is None:
            self.weight_rule = lambda k, i: 1
            self.weighted = False

        # if

        else:
            self.weight_rule = weight_rule
            self.weighted = True

        # else

        # set calibration not calibrated
        self.is_calibrated = False

        # set up the total dataframe
        self.__col_header = [ "type", "angle", "Curvature (1/m)", "Curvature_x (1/m)", "Curvature_y (1/m)", "time" ] + \
                            self.fbg_needle.generate_ch_aa()[ 0 ]
        self.__pred_col_header = [ "AA{} Predicted Curvature_{} (1/m)".format( aa_i, ax ) for aa_i, ax in
                                   itertools.product( range( 1, self.fbg_needle.num_activeAreas + 1 ), [ 'x', 'y' ] ) ]
        self.total_df = pd.DataFrame( columns=self.__col_header )
        self.proc_total_df = self.total_df.copy()  # processed signals
        self.proc_Tcomp_total_df = self.proc_total_df.copy()  # T. compensated processed signals
        self.calval_df = pd.DataFrame( columns=self.__col_header + self.__pred_col_header )  # calibration validation df

        # configure the calibration datasets
        self.calib_dataset = self.configure_dataset( self.calib_directory, self.calib_angles,
                                                     self.calib_curvatures + self.valid_curvatures + [ 0 ] )
        self.valid_dataset = self.configure_dataset( self.valid_directory, self.valid_angles,
                                                     self.calib_curvatures + self.valid_curvatures + [ 0 ] )

    # __init__

    def __write_calibration_errors( self, calib_errors: tuple, output_logfile: str ) -> bool:
        """ Write out the caliration errors to a logfile """
        with open( output_logfile, 'w' ) as logfile:
            logfile.writelines( [ "Needle Calibration log" ] )
            for aa_i in range( 1, self.fbg_needle.num_activeAreas + 1 ):
                delta, residuals, rel_error = calib_errors[ aa_i ]

                logfile.write( f"Active Area {aa_i}\n\n" )

                # compute and write delta error statistics
                min_delta = np.nanmin( delta, axis=0 )
                mean_delta = np.nanmean( delta, axis=0 )
                max_delta = np.nanmax( delta, axis=0 )
                # @formatter:off
                logfile.write( '\n'.join( [
                        "X,Y Reprojection Errors",
                        f"Reprojection errors: {delta}",
                        "Min: ({},{})".format( min_delta[ 0 ], min_delta[ 1 ] ),
                        "Mean: ({},{})".format( mean_delta[ 0 ], mean_delta[ 1 ] ),
                        "Max: ({},{})".format( max_delta[ 0 ], max_delta[ 1 ] )
                        ] ) )
                logfile.write( "\n" + 100*'=' + "\n" )  # add an extra line break
                # @formatter:on

                # compute and write residual error statistics
                min_residual = np.nanmin( residuals )
                mean_residual = np.nanmean( residuals )
                max_residual = np.nanmax( residuals )
                # @formatter:off
                logfile.write( '\n'.join( [
                        "Residual Errors",
                        f"Residuals: {residuals}",
                        f"Min: {min_residual}",
                        f"Mean: {mean_residual}",
                        f"Max: {max_residual}"
                        ] ) )
                logfile.write( "\n" + 100*'=' + "\n" )  # add an extra line break
                # @formatter:on

                # compute and write relative error statistics
                min_rel_error = np.nanmin( rel_error, axis=0 )
                mean_rel_error = np.nanmin( rel_error, axis=0 )
                max_rel_error = np.nanmax( rel_error, axis=0 )
                # @formatter:off
                logfile.write( '\n'.join( [
                        "Relative Errors",
                        f"Relative errors: {rel_error}",
                        f"Min: {min_rel_error}",
                        f"Mean: {mean_rel_error}",
                        f"Max: {max_rel_error}"
                        ] ) )
                logfile.write( "\n" + 100*'=' + "\n" )  # add an extra line break
                # @formatter:on

                logfile.write( "\n" + 100 * '=' + "\n" + 100 * '=' + "\n" )  # add an extra line break
            # for
        # with
        print( f"Wrote calibration error log file: {output_logfile}" )

        return True

    # __write_calibration_errors

    def assign_dataset_type( self, curvatures: list, type: str ):
        """ Assign the dataset type for all of the DataFrames for all of a selected curvature to
            the new type
        """
        # get masks
        mask_func = lambda k: k in curvatures and k != 0  # don't assign a 0-curvature baseline
        mask_calval_df = list( map( mask_func, self.calval_df[ 'Curvature (1/m)' ] ) )
        mask_total_df = list( map( mask_func, self.total_df[ 'Curvature (1/m)' ] ) )
        mask_proc_total_df = list( map( mask_func, self.proc_total_df[ 'Curvature (1/m)' ] ) )
        mask_proc_Tcomp_total_df = list(
                map( mask_func, self.proc_Tcomp_total_df[ 'Curvature (1/m)' ] ) )

        # assign new type
        self.calval_df.loc[ mask_calval_df, 'type' ] = type
        self.total_df.loc[ mask_total_df, 'type' ] = type
        self.proc_total_df.loc[ mask_proc_total_df, 'type' ] = type
        self.proc_Tcomp_total_df.loc[ mask_proc_Tcomp_total_df, 'type' ] = type

    # assign_dataset_type

    @classmethod
    def configure_dataset(cls, directory: str, angles: list, curvatures: list ) -> list:
        """ Configure dataset

            curvatures excluded in order to allow for mapping later

            Returns dataset list of
                [(data directory, experiment angle, curvature (1/m))]


        """
        dataset = [ ]

        if directory is None:
            return dataset

        # if

        # Search for all of the directories
        directories = glob.glob( os.path.join( directory, '*_deg/*/' ) )  # get all of the files

        # iteratre through each to see if it matches the trial directory pattern
        for d in directories:
            res = re.search( cls.directory_pattern, d )

            # make sure that the directory matches
            if res is not None:
                angle, curvature = res.groups()
                angle = float( angle )
                curvature = float( curvature )

                # only include desired curvatures and angles
                # if angle in angles: # process all the curvatures and then remap the angles
                if (angle in angles) and (curvature in curvatures):
                    dataset.append( (d, angle, curvature) )

                # if

            # if
        # for

        return dataset

    # configure_dataset

    def process_dataset( self, dataset: list, type_exp: str, save: bool = True, add_dataset: bool = False ) -> (
            pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """ Process the signals and the datasets 
        
            Args:
                dataset: list of tuples - (data_dir, angle, curvature)
                type_exp: string of experiment type ('calibration' or 'validation' or 'not-assigned')
                save: boolean, whether to save the results to an output file
                add_dataset: whether to add the dataset to the current dataset

        """
        # argument checking
        type_exp_valid_args = [ 'calibration', 'validation', 'not-assigned' ]
        if type_exp not in type_exp_valid_args:
            raise ValueError( "'type_exp' must be in " + str( type_exp_valid_args ) )

        # if

        # AA headers
        ch_aa_head, *_ = self.fbg_needle.generate_ch_aa()

        # instantiate the total dataframe
        total_df = pd.DataFrame( columns=self.total_df.columns )

        # grab all of the data 
        for d, angle, curvature in dataset:
            print( f"Processing directory: {d}" )
            summary_df, _ = combine_fbgdata_directory( d, self.fbg_needle.num_channels, self.fbg_needle.num_activeAreas,
                                                       save=save )

            new_row = summary_df.mean( numeric_only=True ).to_frame().transpose()
            new_row[ 'time' ] = summary_df[ 'time' ].max()
            new_row[ 'angle' ] = 90 - angle  # rotate back into needle coordinate frame
            new_row[ 'Curvature (1/m)' ] = curvature
            new_row[ 'type' ] = type_exp

            # append the dataset
            total_df = pd.concat((total_df, new_row), ignore_index=True, axis=0)

        # for

        # calculate the Curvature_x, Curvature_y
        total_df[ 'Curvature_x (1/m)' ] = total_df[ 'Curvature (1/m)' ] * np.cos(
                np.deg2rad( total_df[ 'angle' ] ) ).round( 10 )
        total_df[ 'Curvature_y (1/m)' ] = total_df[ 'Curvature (1/m)' ] * np.sin(
                np.deg2rad( total_df[ 'angle' ] ) ).round( 10 )

        # process the signals
        proc_total_df, proc_Tcomp_total_df = None, None
        for angle in total_df[ 'angle' ].unique():
            total_angle_df = total_df.loc[ total_df[ 'angle' ] == angle ]
            proc_angle_df = total_angle_df.copy()
            proc_Tcomp_angle_df = proc_angle_df.copy()
            ref_wavelength = total_angle_df.loc[ total_angle_df[ 'Curvature (1/m)' ] == 0, ch_aa_head ].to_numpy()
            signals = total_angle_df[ ch_aa_head ].to_numpy()

            proc_signals = fbg_signal_processing.process_signals( signals, ref_wavelength )
            proc_Tcomp_signals = fbg_signal_processing.temperature_compensation( proc_signals,
                                                                                 self.fbg_needle.num_channels,
                                                                                 self.fbg_needle.num_activeAreas )

            proc_angle_df[ ch_aa_head ] = proc_signals
            proc_Tcomp_angle_df[ ch_aa_head ] = proc_Tcomp_signals

            # add the data to the datasets
            if proc_total_df is None:
                proc_total_df = proc_angle_df

            else:
                proc_total_df = pd.concat((proc_total_df, proc_angle_df), axis=0)

            if proc_Tcomp_total_df is None:
                proc_Tcomp_total_df = proc_Tcomp_angle_df

            else:
                proc_Tcomp_total_df = pd.concat((proc_Tcomp_total_df, proc_Tcomp_angle_df), axis=0)

        # for

        # output the signals
        if save:
            out_dir = os.path.commonpath( [ ds[ 0 ] for ds in dataset ] )
            out_file = os.path.join( out_dir, f'{type_exp}-all-signals.xlsx' )
            with pd.ExcelWriter( out_file ) as xl_writer:
                total_df.to_excel( xl_writer, sheet_name='Raw Signals' )
                proc_total_df.to_excel( xl_writer, sheet_name='Processed Signals' )
                proc_Tcomp_total_df.to_excel( xl_writer, sheet_name='T Comp. Processed Signals' )

            # with

            print( "Wrote all signals to:", out_file )

        # save

        # add the dataset to the current datasets
        if add_dataset:
            self.total_df = self.total_df.append( total_df, ignore_index=True )
            self.proc_total_df = self.proc_total_df.append( proc_total_df, ignore_index=True )
            self.proc_Tcomp_total_df = self.proc_Tcomp_total_df.append( proc_Tcomp_total_df, ignore_index=True )

        # if

        return total_df, proc_total_df, proc_Tcomp_total_df

    # process_dataset

    def run_calibration( self, save: bool = True, fbgneedle_param_outfile: str = None ):
        """ Perform FBG needle calibration procedure

            Args:
                save: boolean (Default: True) on whether to save the new fbgneedle and calibration errors or not.

        """
        # perform the calibration on the calibration dataset (use Tcomp data)
        print( "Calibrating FBG sensors...", end=' ' )
        calibration_df = self.proc_Tcomp_total_df.loc[ self.proc_Tcomp_total_df[ 'type' ] == 'calibration', : ]

        cal_mats, calib_errors = calibrate_sensors_jig( calibration_df, fbg_needle=self.fbg_needle,
                                                        weights_rule=self.weight_rule )

        # update fbg_needle
        self.is_calibrated = True
        self.fbg_needle.cal_matrices = cal_mats
        print( "Calibrated." )
        print()

        # determine predicted curvatures
        curv_prediction = self.fbg_needle.curvatures_processed(
                calibration_df[ self.fbg_needle.generate_ch_aa()[0] ].to_numpy() )
        calibration_df[ self.__pred_col_header ] = curv_prediction.swapaxes( 1, 2 ).reshape( -1,
                                                                                             2 * self.fbg_needle.num_activeAreas )
        self.calval_df = self.calval_df.append( calibration_df, ignore_index=True )  # add to the calval_df

        # save the results
        if save:
            # format the output files
            if fbgneedle_param_outfile is None:
                fbgneedle_param_outfile = "needle_params_{}.json".format(
                        os.path.normpath( self.calib_directory ).split( os.sep )[ -1 ] )

            # if
            else:
                fbgneedle_param_outfile = fbgneedle_param_outfile.format(
                        os.path.normpath( self.calib_directory ).split( os.sep )[ -1 ] )

            # else

            if self.weighted:
                fbgneedle_param_outfile = fbgneedle_param_outfile.replace( '.json', '_weighted.json' )
                self.calval_df[ 'Curvature Weights' ] = [ self.weight_rule( row, idx ) for row, idx in zip(
                        self.calval_df[ [ 'Curvature_x (1/m)', 'Curvature_y (1/m)' ] ].to_numpy(),
                        self.calval_df.index._data ) ]

            # if

            fbgneedle_param_outfile = os.path.join( self.calib_directory, fbgneedle_param_outfile )

            calib_error_logfile = fbgneedle_param_outfile.replace( '.json', '_errors.log' )

            # save FBGNeedle and calibration errors
            self.fbg_needle.save_json( fbgneedle_param_outfile )
            print( f"Saved calibrated needle parameters: {fbgneedle_param_outfile}" )
            self.__write_calibration_errors( calib_errors, calib_error_logfile )
            print()

        # if

        return self.is_calibrated

    # run_calibration

    def run_validation( self, save: bool = True ) -> bool:
        """ Perform FBG needle validation"""
        # perform validation on the validation dataset
        validation_df = self.proc_Tcomp_total_df.loc[ self.proc_Tcomp_total_df[ 'type' ] == 'validation', : ]

        # perform reprojection errors
        print( "Computing errors...", end=' ' )
        curv_prediction = self.fbg_needle.curvatures_processed(
                validation_df[ self.fbg_needle.generate_ch_aa()[ 0 ] ].to_numpy() )
        validation_df.loc[ :, self.__pred_col_header ] = \
            curv_prediction.swapaxes( 1, 2 ).reshape( -1, 2 * self.fbg_needle.num_activeAreas )

        if self.weighted:
            validation_df[ 'Curvature Weights' ] = [ self.weight_rule( row, idx ) for row, idx in zip(
                    validation_df[ [ 'Curvature_x (1/m)', 'Curvature_y (1/m)' ] ].to_numpy(),
                    validation_df.index._data ) ]

        # if

        self.calval_df = self.calval_df.append( validation_df, ignore_index=True )  # add to the calval_df
        print( "Computed." )

        return True

    # run_validation

    def run_calibration_validation( self, save: bool = True, fbgneedle_param_outfile: str = None ):
        """ Calibrate and validate the needle"""
        ret_cal = self.run_calibration( save=save, fbgneedle_param_outfile=fbgneedle_param_outfile )
        ret_val = self.run_validation( save=save )

        return ret_cal, ret_val

    # run_calibration_validation

    def save_processed_data( self, outfile_base: str = '', outdir: str = '' ):
        """ Save the plots and Excel sheets """
        # set-up
        color_scheme = [ 'b', 'g', 'r', 'y', 'c', 'm', 'k' ][ :self.fbg_needle.num_activeAreas ]
        pt_style = [ c + '.' for c in color_scheme ]  # point styles/
        ch_aa_head = self.fbg_needle.generate_ch_aa()[ 0 ]

        # Predicted and actual curvatures
        error_fig, axs = plt.subplots( nrows=2, ncols=self.fbg_needle.num_activeAreas )
        error_fig.set_size_inches( [ 15, 8 ] )
        curv_fig, curv_axs = plt.subplots( nrows=2, ncols=self.fbg_needle.num_activeAreas)
        sq_style = [ 's' + c for c in color_scheme ]
        for aa_i in range( 1, self.fbg_needle.num_activeAreas + 1 ):
            self.calval_df[ f'AA{aa_i} Error Curvature_x (1/m)' ] = \
                self.calval_df[ f'AA{aa_i} Predicted Curvature_x (1/m)' ] - self.calval_df[ 'Curvature_x (1/m)' ]
            self.calval_df[ f'AA{aa_i} Error Curvature_y (1/m)' ] = \
                self.calval_df[ f'AA{aa_i} Predicted Curvature_y (1/m)' ] - self.calval_df[ 'Curvature_y (1/m)' ]

            # plot calibration errors
            self.calval_df.loc[ self.calval_df[ 'type' ] == 'calibration' ].plot( x='Curvature_x (1/m)',
                                                                                  y=f'AA{aa_i} Error Curvature_x (1/m)',
                                                                                  ax=axs[ 0, aa_i - 1 ],
                                                                                  style='.',
                                                                                  xlabel='Curvature x (1/m)',
                                                                                  ylabel='Error in Curvature x (1/m)',
                                                                                  title=f'Error of AA{aa_i}' )

            self.calval_df.loc[ self.calval_df[ 'type' ] == 'calibration' ].plot( x='Curvature_y (1/m)',
                                                                                  y=f'AA{aa_i} Error Curvature_y (1/m)',
                                                                                  ax=axs[ 1, aa_i - 1 ],
                                                                                  style='.',
                                                                                  xlabel='Curvature y (1/m)',
                                                                                  ylabel='Error in Curvature y (1/m)' )

            # plot validation errors
            self.calval_df.loc[ self.calval_df[ 'type' ] == 'validation' ].plot( x='Curvature_x (1/m)',
                                                                                 y=f'AA{aa_i} Error Curvature_x (1/m)',
                                                                                 ax=axs[ 0, aa_i - 1 ],
                                                                                 style='s',
                                                                                 xlabel='Curvature x (1/m)',
                                                                                 ylabel='Error in Curvature x (1/m)' )

            self.calval_df.loc[ self.calval_df[ 'type' ] == 'validation' ].plot( x='Curvature_y (1/m)',
                                                                                 y=f'AA{aa_i} Error Curvature_y (1/m)',
                                                                                 ax=axs[ 1, aa_i - 1 ],
                                                                                 style='s',
                                                                                 xlabel='Curvature y (1/m)',
                                                                                 ylabel='Error in Curvature y (1/m)' )
            
            # plot calibration lines
            self.calval_df.loc[ self.calval_df[ 'type' ] == 'calibration' ].plot( x='Curvature_x (1/m)',
                                                                                  y=f'AA{aa_i} Predicted Curvature_x (1/m)',
                                                                                  ax=curv_axs[ 0, aa_i - 1 ],
                                                                                  style='.',
                                                                                  xlabel='Curvature x (1/m)',
                                                                                  ylabel='Predicted Curvature x (1/m)',
                                                                                  title=f'Curvature of AA{aa_i}' )

            self.calval_df.loc[ self.calval_df[ 'type' ] == 'calibration' ].plot( x='Curvature_y (1/m)',
                                                                                  y=f'AA{aa_i} Predicted Curvature_y (1/m)',
                                                                                  ax=curv_axs[ 1, aa_i - 1 ],
                                                                                  style='.',
                                                                                  xlabel='Curvature y (1/m)',
                                                                                  ylabel='Predicted Curvature y (1/m)' )

            # plot validation lines
            self.calval_df.loc[ self.calval_df[ 'type' ] == 'validation' ].plot( x='Curvature_x (1/m)',
                                                                                 y=f'AA{aa_i} Predicted Curvature_x (1/m)',
                                                                                 ax=curv_axs[ 0, aa_i - 1 ],
                                                                                 style='s',
                                                                                 xlabel='Curvature x (1/m)',
                                                                                 ylabel='Predicted Curvature x (1/m)' )

            self.calval_df.loc[ self.calval_df[ 'type' ] == 'validation' ].plot( x='Curvature_y (1/m)',
                                                                                 y=f'AA{aa_i} Predicted Curvature_y (1/m)',
                                                                                 ax=curv_axs[ 1, aa_i - 1 ],
                                                                                 style='s',
                                                                                 xlabel='Curvature y (1/m)',
                                                                                 ylabel='Predicted Curvature y (1/m)' )
            
            # plot true lines  y=x
            kx = np.linspace(
                self.calval_df['Curvature_x'].min(), 
                self.calval_df['Curvature_x'].max(),
                100
            )
            ky = np.linspace(
                self.calval_df['Curvature_y'].min(), 
                self.calval_df['Curvature_y'].max(),
                100
            )
            curv_axs[0, aa_i-1].plot(kx, kx, 'r--', label='true')
            curv_axs[1, aa_i-1].plot(ky, ky, 'r--', label='true')

            # format the axes
            axs[ 0, aa_i - 1 ].legend( [ 'calibration', 'validation' ] )
            axs[ 1, aa_i - 1 ].legend( [ 'calibration', 'validation' ] )
            curv_axs[ 0, aa_i - 1 ].legend( [ 'calibration', 'validation' ] )
            curv_axs[ 1, aa_i - 1 ].legend( [ 'calibration', 'validation' ] )

        # for
        error_fig.suptitle( "Predicted curvature error" )
        outfile_fig_error = os.path.join( outdir,
                                          f"{outfile_base}all-curvature-error.png" )
        outfile_fig_curv = os.path.join( outdir,
                                         f"{outfile_base}all-curvatures.png")
        if self.weighted:
            outfile_fig_error = outfile_fig_error.replace( '.png', '_weighted.png' )
            outfile_fig_curv  = outfile_fig_curv.replace( '.png', '_weighted.png' )
        
        # if

        error_fig.savefig( outfile_fig_error )
        print( "Saved figure: {}".format( outfile_fig_error ) )
        curv_fig.savefig( outfile_fig_curv )
        print( "Saved figure: {}".format( outfile_fig_curv ) )

        # Output the Excel sheets of the data
        data_outfile = f"{outfile_base}Jig-Calibration-Validation-Data.xlsx"
        data_outfile = os.path.join( outdir, data_outfile )
        if self.weighted:
            data_outfile = data_outfile.replace( '.xlsx', '_weighted.xlsx' )

        with pd.ExcelWriter( data_outfile ) as xl_writer:
            self.total_df.to_excel( xl_writer, sheet_name='Raw Signals' )
            self.proc_total_df.to_excel( xl_writer, sheet_name='Processed Signals' )
            self.proc_Tcomp_total_df.to_excel( xl_writer, sheet_name='T Comp. Processed Signals' )
            self.calval_df.to_excel( xl_writer, sheet_name='Calibration Validation Dataset' )

        # with
        print( f"Saved data file: {data_outfile}" )

        # Output the analytical figures of the signals
        for exp_type in self.total_df[ 'type' ].unique():
            # grab the specific dataset
            sub_total_df = self.total_df.loc[ self.total_df[ 'type' ] == exp_type, : ]
            sub_proc_total_df = self.proc_total_df.loc[ self.proc_total_df[ 'type' ] == exp_type, : ]
            sub_proc_Tcomp_total_df = self.proc_Tcomp_total_df.loc[ self.proc_Tcomp_total_df[ 'type' ] == exp_type, : ]

            # Plot signals vs. Raw Curvature per angle
            for angle in sub_total_df[ 'angle' ].unique():
                fig_raw, axs_raw = plt.subplots( self.fbg_needle.num_activeAreas, sharex='col' )
                fig_raw.set_size_inches( [ 13, 8 ] )
                for aa_i in range( 1, self.fbg_needle.num_activeAreas + 1 ):
                    ch_aa_i = list( filter( lambda ca: f"AA{aa_i}" in ca, ch_aa_head ) )
                    head_aa_i = [ "Curvature (1/m)" ] + ch_aa_i

                    # plot the raw signals
                    sub_total_df.loc[ sub_total_df[ 'angle' ] == angle, head_aa_i ].plot( x='Curvature (1/m)',
                                                                                          y=ch_aa_i,
                                                                                          style=pt_style,
                                                                                          ax=axs_raw[ aa_i - 1 ] )

                    # plot the linear fits for each channel
                    line_colors = { c.get_label(): c.get_color() for c in axs_raw[ aa_i - 1 ].get_children() if
                                    isinstance( c, Line2D ) }
                    for ch_i in range( 1, self.fbg_needle.num_channels + 1 ):
                        # linear fit
                        ch_i_aa_i = "CH{0} | AA{1}".format( ch_i, aa_i )
                        m, b, R_sq = _linear_fit(
                                sub_total_df.loc[ sub_total_df[ 'angle' ] == angle, 'Curvature (1/m)' ],
                                sub_total_df.loc[ sub_total_df[ 'angle' ] == angle, ch_i_aa_i ] )
                        k_fit = np.linspace(
                                sub_total_df.loc[ sub_total_df[ 'angle' ] == angle, 'Curvature (1/m)' ].min(),
                                sub_total_df.loc[ sub_total_df[ 'angle' ] == angle, 'Curvature (1/m)' ].max(), 100 )
                        sig_fit = m * k_fit + b

                        axs_raw[ aa_i - 1 ].plot( k_fit, sig_fit, color=line_colors[ ch_i_aa_i ],
                                                  label=ch_i_aa_i + " linear" )

                    # for: channels

                    axs_raw[ aa_i - 1 ].legend()  # turn on the legend (if not already)

                # for: active areas

                # save and format the figure
                axs_raw[ -1 ].set_xlabel( 'Curvature (1/m)' )
                fig_raw.suptitle( f"{angle} deg: Raw Signals vs. Curvature" )
                outfile_fig_raw = os.path.join( outdir,
                                                f"{outfile_base}{angle}_deg_{exp_type}_aa-signals-curvature.png" )
                if self.weighted:
                    outfile_fig_raw = outfile_fig_raw.replace( '.png', '_weighted.png' )

                fig_raw.savefig( outfile_fig_raw )
                print( "Saved figure:", outfile_fig_raw )
                plt.close( fig=fig_raw )

            # for: angles

            print()

            # Plot Processed Signals vs. Curvature_x, Curvature_y
            fig_proc, axs_proc = plt.subplots( nrows=self.fbg_needle.num_activeAreas, ncols=2, sharex='col' )
            fig_proc.set_size_inches( [ 14, 9 ] )
            for aa_i in range( 1, self.fbg_needle.num_activeAreas + 1 ):
                ch_aa_i = list( filter( lambda ca: f"AA{aa_i}" in ca, ch_aa_head ) )
                head_aa_i = [ 'Curvature_x (1/m)', 'Curvature_y (1/m)' ] + ch_aa_i

                # plot the processed signals
                sub_proc_total_df[ head_aa_i ].plot( x='Curvature_x (1/m)', y=ch_aa_i, style=pt_style,
                                                     ax=axs_proc[ aa_i - 1, 0 ] )
                sub_proc_total_df[ head_aa_i ].plot( x='Curvature_y (1/m)', y=ch_aa_i, style=pt_style,
                                                     ax=axs_proc[ aa_i - 1, 1 ] )

                # fit lines to processed signals
                line_colors = { c.get_label(): c.get_color() for c in axs_proc[ aa_i - 1, 0 ].get_children() if
                                isinstance( c, Line2D ) }
                for ch_i in range( 1, self.fbg_needle.num_channels + 1 ):
                    ch_i_aa_i = "CH{0} | AA{1}".format( ch_i, aa_i )
                    # get x linear fit
                    mx, bx, R_sqx = _linear_fit( sub_proc_total_df[ 'Curvature_x (1/m)' ],
                                                 sub_proc_total_df[ ch_i_aa_i ] )
                    kx_fit = np.linspace( sub_proc_total_df[ 'Curvature_x (1/m)' ].min(),
                                          sub_proc_total_df[ 'Curvature_x (1/m)' ].max(), 100 )
                    sigx_fit = mx * kx_fit + bx

                    # get y linear fit
                    my, by, R_sqy = _linear_fit( sub_proc_total_df[ 'Curvature_y (1/m)' ],
                                                 sub_proc_total_df[ ch_i_aa_i ] )
                    ky_fit = np.linspace( sub_proc_total_df[ 'Curvature_y (1/m)' ].min(),
                                          sub_proc_total_df[ 'Curvature_y (1/m)' ].max(), 100 )
                    sigy_fit = my * ky_fit + by

                    # plot the linear fits
                    axs_proc[ aa_i - 1, 0 ].plot( kx_fit, sigx_fit, color=line_colors[ ch_i_aa_i ],
                                                  label=ch_i_aa_i + " linear" )
                    axs_proc[ aa_i - 1, 1 ].plot( ky_fit, sigy_fit, color=line_colors[ ch_i_aa_i ],
                                                  label=ch_i_aa_i + " linear" )
                # for: channels
            # for: active areas

            # plot formatting and saving
            fig_proc.suptitle( 'Signal Shifts vs Curvature (X | Y)' )
            axs_proc[ -1, 0 ].set_xlabel( 'Curvature X (1/m)' )  # set x axis labels
            axs_proc[ -1, 1 ].set_xlabel( 'Curvature Y (1/m)' )

            outfile_fig_proc = os.path.join( outdir, f"{outfile_base}{exp_type}_all-curvatures-xy_signal-shifts.png" )
            if self.weighted:
                outfile_fig_proc = outfile_fig_proc.replace( '.png', '_weighted.png' )

            fig_proc.savefig( outfile_fig_proc )
            print( "Saved figure:", outfile_fig_proc )
            plt.close( fig=fig_proc )

            # Plot T. Compensated Processed Signals vs. Curvature_x, Curvature_y
            fig_proc_Tcomp, axs_proc_Tcomp = plt.subplots( nrows=self.fbg_needle.num_activeAreas, ncols=2,
                                                           sharex='col' )
            fig_proc_Tcomp.set_size_inches( [ 14, 9 ] )
            for aa_i in range( 1, self.fbg_needle.num_activeAreas + 1 ):
                ch_aa_i = list( filter( lambda ca: f"AA{aa_i}" in ca, ch_aa_head ) )
                head_aa_i = [ 'Curvature_x (1/m)', 'Curvature_y (1/m)' ] + ch_aa_i

                # plot the proc_Tcompessed signals
                sub_proc_Tcomp_total_df[ head_aa_i ].plot( x='Curvature_x (1/m)', y=ch_aa_i, style=pt_style,
                                                           ax=axs_proc_Tcomp[ aa_i - 1, 0 ] )
                sub_proc_Tcomp_total_df[ head_aa_i ].plot( x='Curvature_y (1/m)', y=ch_aa_i, style=pt_style,
                                                           ax=axs_proc_Tcomp[ aa_i - 1, 1 ] )

                # fit lines to proc_Tcompessed signals
                line_colors = { c.get_label(): c.get_color() for c in axs_proc_Tcomp[ aa_i - 1, 0 ].get_children() if
                                isinstance( c, Line2D ) }
                for ch_i in range( 1, self.fbg_needle.num_channels + 1 ):
                    ch_i_aa_i = "CH{0} | AA{1}".format( ch_i, aa_i )
                    # get x linear fit
                    mx, bx, R_sqx = _linear_fit( sub_proc_Tcomp_total_df[ 'Curvature_x (1/m)' ],
                                                 sub_proc_Tcomp_total_df[ ch_i_aa_i ] )
                    kx_fit = np.linspace( sub_proc_Tcomp_total_df[ 'Curvature_x (1/m)' ].min(),
                                          sub_proc_Tcomp_total_df[ 'Curvature_x (1/m)' ].max(), 100 )
                    sigx_fit = mx * kx_fit + bx

                    # get y linear fit
                    my, by, R_sqy = _linear_fit( sub_proc_Tcomp_total_df[ 'Curvature_y (1/m)' ],
                                                 sub_proc_Tcomp_total_df[ ch_i_aa_i ] )
                    ky_fit = np.linspace( sub_proc_Tcomp_total_df[ 'Curvature_y (1/m)' ].min(),
                                          sub_proc_Tcomp_total_df[ 'Curvature_y (1/m)' ].max(), 100 )
                    sigy_fit = my * ky_fit + by

                    # plot the linear fits
                    axs_proc_Tcomp[ aa_i - 1, 0 ].plot( kx_fit, sigx_fit, color=line_colors[ ch_i_aa_i ],
                                                        label=ch_i_aa_i + " linear" )
                    axs_proc_Tcomp[ aa_i - 1, 1 ].plot( ky_fit, sigy_fit, color=line_colors[ ch_i_aa_i ],
                                                        label=ch_i_aa_i + " linear" )
                # for: channels
            # for: active areas

            # plot formatting and saving
            fig_proc_Tcomp.suptitle( 'Signal Shifts vs Curvature (X | Y)' )
            axs_proc_Tcomp[ -1, 0 ].set_xlabel( 'Curvature X (1/m)' )  # set x axis labels
            axs_proc_Tcomp[ -1, 1 ].set_xlabel( 'Curvature Y (1/m)' )

            outfile_fig_proc_Tcomp = os.path.join( outdir,
                                                   f"{outfile_base}{exp_type}_all-curvatures-xy_signal-shifts-T-comp.png" )
            if self.weighted:
                outfile_fig_proc_Tcomp = outfile_fig_proc_Tcomp.replace( '.png', '_weighted.png' )

            fig_proc_Tcomp.savefig( outfile_fig_proc_Tcomp )
            print( "Saved figure:", outfile_fig_proc_Tcomp )
            plt.close( fig=fig_proc_Tcomp )

        # for: types of experiment

    # save_processed_data


# class: FBGNeedleJigCalibrator


def _linear_fit( x, y ) -> (float, float, float):
    """ Perform a linear fit and return the R^2 value

        Args:
            x: the x-coordinates
            y: the y-coordinates

        Return:
            slope, y-intercept, R^2 coefficient

    """
    if len( np.unique( x ) ) < 2:
        coeffs = (0, np.polyfit( x, y, 0, rcond=None )[ 0 ])
        R_sq = 1

    # if
    else:
        coeffs = np.polyfit( x, y, 1, rcond=None )
        R_sq = np.corrcoef( x, y )[ 0, 1 ] ** 2  # R^2

    # else

    return coeffs[ 0 ], coeffs[ 1 ], R_sq


# _linear_fit

def _leastsq_fit( signals: np.ndarray, curvatures: np.ndarray, weight_rule: callable ):
    """ Perform least squares fitting of curvatures
        solves
        argmin_C ||Weights^(1/2).(C.signals - curvatures)||^2

        Args:
            signals: numpy array of shape (N, num_active_areas)
            curvatures: numpy array of shape (N, 2) (curvature_x, curvature_x)
            weight_rule: function for weighting rule of 2 args (curvature_xy_row, row_index)
        Return:
            calibration matrix, numpy array of shape (num_active_areas, 2)
            errors, tuple of errors: (reprojection errors, residuals, relative errors)

    """
    # size checking
    assert (signals.shape[ 0 ] == curvatures.shape[ 0 ])  # same number of data samples
    assert (curvatures.shape[ 1 ] == 2)  # 2 curvature values only

    # build weighting matrix
    weights = np.sqrt( np.diag( [ weight_rule( curv_row, idx ) for idx, curv_row in enumerate( curvatures ) ] ) )

    signals_w = weights @ signals
    curvatures_w = weights @ curvatures

    C, _, rnk, sng = np.linalg.lstsq( signals_w, curvatures_w, rcond=None )

    # compute the errors
    delta = curvatures - signals @ C
    residuals = np.linalg.norm( delta, axis=1 )
    rel_errors = delta / np.linalg.norm( curvatures )
    rel_errors[ residuals == 0 ] = 0  # make sure all 0's are accounted for
    rel_errors[ np.isinf( rel_errors ) ] = np.nan

    errors = (delta, residuals, rel_errors)

    return C.T, errors


# _leastsq_fit

def calibrate_sensors_jig( *args, **kwargs ) -> (dict, dict):
    """ Actually calibrate the FBG sensors per AA

        Args:
            proc_df: pandas DataFrame of the entire processed data
            curvatures: dict {aa_idx: np.array([curvature_x, curvature_y])
            signals: dict {aa_idx: np.array([signal_aa1, ..., signal_aan])
            num_channels: int (Default None)
            num_active_areas: int (Default None)
            fbg_needle: FBGNeedle (Default None)
            weights: np.array of shape (curvatures.shape[0])
            weights_rule: callable to determine the weights call(curvature_row, index)

        Return:
            calibration matrices, dict: {aa_idx: cal_mat} cal_mat = 2 x num_active_areas
            calibration errors,   dict: {aa_idx: cal_error} error = (delta, residuals, relative errors)

    """
    # parse FBG param args
    if 'fbg_needle' in kwargs.keys():
        fbg_needle = kwargs[ 'fbg_needle' ]
        assert (isinstance( fbg_needle, FBGNeedle ))
        num_channels = fbg_needle.num_channels
        num_active_areas = fbg_needle.num_activeAreas

        ch_aa_names, *_ = fbg_needle.generate_ch_aa()
        aa_assignments = np.array( fbg_needle.assignments_aa() )
        if isinstance(fbg_needle, MCFNeedle):
            mask_centralcore = fbg_needle.assignments_centralcore()
            ch_aa_names = [
                    ch_aa
                    for ch_aa, is_central_ch in zip(ch_aa_names, mask_centralcore)
                    if not is_central_ch
            ]
            aa_assignments = aa_assignments[np.logical_not(mask_centralcore)]
        #

    # if

    elif all( k in kwargs.keys() for k in [ 'num_channels', 'num_active_areas' ] ):
        num_channels = kwargs[ 'num_channels' ]
        num_active_areas = kwargs[ 'num_active_areas' ]

        ch_aa_names, = FBGNeedle.generate_ch_aa( num_channels, num_active_areas )
        aa_assignments = np.array( FBGNeedle.assignments_aa( num_channels, num_active_areas ) )

        # value and type checking
        assert (isinstance( num_channels, int ) and num_channels > 0)
        assert (isinstance( num_active_areas, int ) and num_active_areas > 0)

    # elif

    else:
        raise ValueError( "kwarg needed is either 'fbg_needle' or ('num_channels', 'num_active_areas'" )

    # else

    # parse data args
    if len( args ) == 1:
        proc_df = args[ 0 ]
        assert (isinstance( proc_df, pd.DataFrame ))

        # grab the relevant values
        curvatures_single = proc_df[ [ 'Curvature_x (1/m)', 'Curvature_y (1/m)' ] ]
        signals_tot = proc_df[ ch_aa_names ]

        # turn the values into a dict of numpy array's
        curvatures = { aa_i: curvatures_single.to_numpy() for aa_i in range( 1, num_active_areas + 1 ) }
        signals = { aa_i: signals_tot.loc[ :, aa_assignments == aa_i ].to_numpy() for aa_i in
                    range( 1, num_active_areas + 1 ) }

    # if

    elif len( args ) == 2:
        curvatures = args[ 0 ]
        signals = args[ 1 ]

        assert (isinstance( curvatures, dict ))  # type checking
        assert (isinstance( signals, dict ))

    # elif

    else:
        raise IndexError( "Input should be either (pandas.DataFrame) or (dict, dict)." )

    # else

    # parse the weighting rule
    if 'weights' in kwargs.keys():
        weights = kwargs[ 'weights' ]
        assert (isinstance( weights, np.ndarray ))
        weights = weights.flatten()  # ensure that it is a flat array
        weights_rule = lambda k, i: weights[ i ]  # curvature, index

    # if
    elif 'weights_rule' in kwargs.keys():
        weights_rule = kwargs[ 'weights_rule' ]
        assert (callable( weights_rule ))

    # elif

    else:  # default weighting
        weights_rule = lambda k, i: 1

    # else

    # perform least squares calibration
    cal_mats = { }
    calib_errors = { }
    for aa_i in range( 1, num_active_areas + 1 ):
        # get the curvature and signal values from the dicts
        curvature_aa_i = curvatures[ aa_i ]
        signals_aa_i = signals[ aa_i ]

        # determine the calibration matrix
        cal_mat_aa_i, errors_aa_i = _leastsq_fit( signals_aa_i, curvature_aa_i, weights_rule )

        # append the values for calibration matrix and errors
        cal_mats[ aa_i ] = cal_mat_aa_i
        calib_errors[ aa_i ] = errors_aa_i

    # for

    return cal_mats, calib_errors


# calibrate_sensors_jig

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
        new_row = df.mean( numeric_only=True ).to_frame().transpose()  # mean peak values
        new_row[ 'time' ] = df[ 'time' ].max()  # latest time

        summary_df = pd.concat((summary_df, new_row), ignore_index=True)

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
                             outfile: str = None ) -> (pd.DataFrame, pd.DataFrame):
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

def jig_calibration( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list,
                     calib_error_logfile: str = None, weights_rule: callable = lambda k, i: 1 ) -> FBGNeedle:
    """ Perform jig calibration """
    warnings.warn( DeprecationWarning( 'Use class implementation of jig calibration.' ) )
    # process the FBG signals
    total_df, _, proc_Tcomp_total_df = jig_process_signals( directory, dirs_degs, fbg_needle, curvatures_list )

    # perform calibration via least squares formulation
    cal_mats, calib_errors = calibrate_sensors_jig( proc_Tcomp_total_df, fbg_needle=fbg_needle,
                                                    weights_rule=weights_rule
                                                    )

    # output results to a log file
    if calib_error_logfile is not None:
        with open( calib_error_logfile, 'w' ) as logfile:
            logfile.writelines( [ "Needle Calibration log" ] )
            for aa_i in range( 1, fbg_needle.num_activeAreas + 1 ):
                delta, residuals, rel_error = calib_errors[ aa_i ]

                logfile.write( f"Active Area {aa_i}\n\n" )

                # compute and write delta error statistics
                min_delta = np.nanmin( delta, axis=0 )
                mean_delta = np.nanmean( delta, axis=0 )
                max_delta = np.nanmax( delta, axis=0 )
                # @formatter:off
                logfile.write( '\n'.join( [
                        "X,Y Reprojection Errors",
                        f"Reprojection errors: {delta}",
                        "Min: ({},{})".format( min_delta[ 0 ], min_delta[ 1 ] ),
                        "Mean: ({},{})".format( mean_delta[ 0 ], mean_delta[ 1 ] ),
                        "Max: ({},{})".format( max_delta[ 0 ], max_delta[ 1 ] )
                        ] ) )
                logfile.write( "\n" + 100*'=' + "\n" )  # add an extra line break
                # @formatter:on

                # compute and write residual error statistics
                min_residual = np.nanmin( residuals )
                mean_residual = np.nanmean( residuals )
                max_residual = np.nanmax( residuals )
                # @formatter:off
                logfile.write( '\n'.join( [
                        "Residual Errors",
                        f"Residuals: {residuals}",
                        f"Min: {min_residual}",
                        f"Mean: {mean_residual}",
                        f"Max: {max_residual}"
                        ] ) )
                logfile.write( "\n" + 100*'=' + "\n" )  # add an extra line break
                # @formatter:on

                # compute and write relative error statistics
                min_rel_error = np.nanmin( rel_error, axis=0 )
                mean_rel_error = np.nanmin( rel_error, axis=0 )
                max_rel_error = np.nanmax( rel_error, axis=0 )
                # @formatter:off
                logfile.write( '\n'.join( [
                        "Relative Errors",
                        f"Relative errors: {rel_error}",
                        f"Min: {min_rel_error}",
                        f"Mean: {mean_rel_error}",
                        f"Max: {max_rel_error}"
                        ] ) )
                logfile.write( "\n" + 100*'=' + "\n" )  # add an extra line break
                # @formatter:on

                logfile.write( "\n" + 100 * '=' + "\n" + 100 * '=' + "\n" )  # add an extra line break
            # for
        # with
        print( f"Wrote calibration error log file: {calib_error_logfile}" )
    # if

    # update the calibrated fbg_needle
    fbg_needle.set_calibration_matrices( cal_mats )

    return fbg_needle


# jig_calibration

def jig_process_signals( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list,
                         outfile_base: str = "" ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """ Process the FBG signals """
    # set-up
    aa_assignments = np.array( fbg_needle.assignments_aa() )
    ch_assignments = np.array( fbg_needle.assignments_ch() )
    ch_aa_name, *_ = fbg_needle.generate_ch_aa()

    # process all of the FBG data files
    for d in sum( dirs_degs.values(), [ ] ):
        if os.path.isdir( d ):
            print( "Processing FBG data directory:", d )
        if os.path.exists( os.path.join( d, 'fbgdata.xlsx' ) ):
            print( "Already processed. Continuing..." )

        # if
        else:
            combine_fbgdata_directory( d, fbg_needle.num_channels, fbg_needle.num_activeAreas, save=True )
            print( 'Completed FBG data directory:', d )

        # else
        print()

    # for

    # iterate through each of the angles to get the FBG data files to summarize the curvature trials
    total_df = None
    header = [ 'angle', 'Curvature (1/m)', 'time' ] + ch_aa_name
    for angle, fbgdata_dir in dirs_degs.items():
        fbgdata_files = [ os.path.join( d, 'fbgdata.xlsx' ) for d in fbgdata_dir ]
        outfile_fbgbase = os.path.join( directory, f"{outfile_base}FBGResults_{angle}_deg" )
        out_fbgresult_file = outfile_fbgbase + ".xlsx"

        # consolidate the FBG data directories into one FBG file
        summary_df, *_ = combine_fbgdata_summary( fbgdata_files, curvatures_list, fbg_needle.num_channels,
                                                  fbg_needle.num_activeAreas, out_fbgresult_file
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
        fig, axs = plt.subplots( fbg_needle.num_activeAreas, sharex='col' )
        fig.set_size_inches( [ 13, 8 ] )
        for aa_i in range( 1, fbg_needle.num_activeAreas + 1 ):
            mask_signals = (aa_i == aa_assignments)  # determine which columns of the FBG signals to get
            mask = np.append( [ False, True, False ], mask_signals )  # include the curvatures and remove time

            summary_df.loc[ :, mask ].plot( x='Curvature (1/m)', style='.', ax=axs[ aa_i - 1 ] )
            axs[ aa_i - 1 ].set_ylabel( 'signal (nm)' )

            # plot linear fits
            line_colors = { c.get_label(): c.get_color() for c in axs[ aa_i - 1 ].get_children() if
                            isinstance( c, Line2D ) }
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
        outfile_fig = os.path.join( outpath, f"{outfile_base}{angle}_deg_aa-signals-curvature.png" )
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
                                                                              fbg_needle.num_activeAreas )

        # change the values for the signals
        proc_total_df.loc[ proc_total_df[ 'angle' ] == angle, ch_aa_name ] = signal_shifts
        proc_Tcomp_total_df.loc[ proc_Tcomp_total_df[ 'angle' ] == angle, ch_aa_name ] = Tcomp_signal_shifts

    # for

    # save the tables
    outfile_total_df = os.path.join( directory, f'{outfile_base}all-curvatures-signals.xlsx' )
    with pd.ExcelWriter( outfile_total_df ) as xl_writer:
        total_df.to_excel( xl_writer, sheet_name='Raw Signals' )
        proc_total_df.to_excel( xl_writer, sheet_name='Processed Signals' )
        proc_Tcomp_total_df.to_excel( xl_writer, sheet_name='T Comp. Processed Signals' )

    # with
    print( f"Saved data file: {outfile_total_df}" )

    # Plot linear fits wavelength shift vs curvature_x and curvature_y
    color_scheme = [ 'b', 'g', 'r', 'y', 'c', 'm', 'k' ][ :fbg_needle.num_activeAreas ]
    pt_style = [ c + '.' for c in color_scheme ]  # point styles

    fig_proc, axs_proc = plt.subplots( nrows=fbg_needle.num_activeAreas, ncols=2, sharex='col' )
    fig_proc.set_size_inches( [ 14, 9 ] )

    for aa_i in range( 1, fbg_needle.num_activeAreas + 1 ):
        mask_signals = (aa_assignments == aa_i)
        mask_x = np.append( [ False, False, True, False, False ], mask_signals )  # curvature_x and AA signals
        mask_y = np.append( [ False, False, False, True, False ], mask_signals )  # curvature and AA signals

        # plot the processed signals
        proc_total_df.loc[ :, mask_x ].plot( x='Curvature_x (1/m)', style=pt_style, ax=axs_proc[ aa_i - 1, 0 ] )
        proc_total_df.loc[ :, mask_y ].plot( x='Curvature_y (1/m)', style=pt_style, ax=axs_proc[ aa_i - 1, 1 ] )

        # fit lines to the processed signals
        line_colors = { c.get_label(): c.get_color() for c in axs_proc[ aa_i - 1, 0 ].get_children() if
                        isinstance( c, Line2D ) }
        for ch_aa_idx in np.where( np.append( 5 * [ False ], mask_signals ) )[ 0 ]:
            # get the x linear fit
            mx, bx, R_sqx = _linear_fit( proc_total_df[ 'Curvature_x (1/m)' ], proc_total_df.iloc[ :, ch_aa_idx ] )
            kx = np.linspace( proc_total_df[ 'Curvature_x (1/m)' ].min(), proc_total_df[ 'Curvature_x (1/m)' ].max(),
                              100 )
            sigx = mx * kx + bx

            # get the y linear fit
            my, by, R_sqy = _linear_fit( proc_total_df[ 'Curvature_y (1/m)' ], proc_total_df.iloc[ :, ch_aa_idx ] )
            ky = np.linspace( proc_total_df[ 'Curvature_y (1/m)' ].min(), proc_total_df[ 'Curvature_y (1/m)' ].max(),
                              100 )
            sigy = my * ky + by

            # get the current plot color
            ch_aa = proc_total_df.columns[ ch_aa_idx ]
            color = line_colors[ ch_aa ]

            # plot the linear fit
            axs_proc[ aa_i - 1, 0 ].plot( kx, sigx, color=color, label=ch_aa + " linear" )
            axs_proc[ aa_i - 1, 1 ].plot( ky, sigy, color=color, label=ch_aa + " linear" )

        # for
        axs_proc[ aa_i - 1, 0 ].legend()  # turn on the legends
        axs_proc[ aa_i - 1, 1 ].legend()
        axs_proc[ aa_i - 1, 0 ].set_ylabel( 'Signal Shifts (nm)' )  # set y-axis labels

    # for

    # plot formatting
    fig_proc.suptitle( 'Signal Shifts vs Curvature (X | Y)' )
    axs_proc[ -1, 0 ].set_xlabel( 'Curvature X (1/m)' )  # set x axis labels
    axs_proc[ -1, 1 ].set_xlabel( 'Curvature Y (1/m)' )

    # plot the T compensated processed signals
    fig_proc_Tcomp, axs_proc_Tcomp = plt.subplots( nrows=fbg_needle.num_activeAreas, ncols=2, sharex='col' )
    fig_proc_Tcomp.set_size_inches( [ 14, 9 ] )
    for aa_i in range( 1, fbg_needle.num_activeAreas + 1 ):
        mask_signals = (aa_assignments == aa_i)
        mask_x = np.append( [ False, False, True, False, False ], mask_signals )  # curvature_x and AA signals
        mask_y = np.append( [ False, False, False, True, False ], mask_signals )  # curvature and AA signals

        # plot the processed signals
        proc_Tcomp_total_df.loc[ :, mask_x ].plot( x='Curvature_x (1/m)', style=pt_style,
                                                   ax=axs_proc_Tcomp[ aa_i - 1, 0 ] )
        proc_Tcomp_total_df.loc[ :, mask_y ].plot( x='Curvature_y (1/m)', style=pt_style,
                                                   ax=axs_proc_Tcomp[ aa_i - 1, 1 ] )

        # fit lines to the processed signals
        line_colors = { c.get_label(): c.get_color() for c in axs_proc_Tcomp[ aa_i - 1, 0 ].get_children() if
                        isinstance( c, Line2D ) }
        for ch_aa_idx in np.where( np.append( 5 * [ False ], mask_signals ) )[ 0 ]:
            # get the x linear fit
            mx, bx, R_sqx = _linear_fit( proc_Tcomp_total_df[ 'Curvature_x (1/m)' ],
                                         proc_Tcomp_total_df.iloc[ :, ch_aa_idx ] )
            kx = np.linspace( proc_Tcomp_total_df[ 'Curvature_x (1/m)' ].min(),
                              proc_Tcomp_total_df[ 'Curvature_x (1/m)' ].max(),
                              100 )
            sigx = mx * kx + bx

            # get the y linear fit
            my, by, R_sqy = _linear_fit( proc_Tcomp_total_df[ 'Curvature_y (1/m)' ],
                                         proc_Tcomp_total_df.iloc[ :, ch_aa_idx ] )
            ky = np.linspace( proc_Tcomp_total_df[ 'Curvature_y (1/m)' ].min(),
                              proc_Tcomp_total_df[ 'Curvature_y (1/m)' ].max(),
                              100 )
            sigy = my * ky + by

            # get the current plot color
            ch_aa = proc_Tcomp_total_df.columns[ ch_aa_idx ]
            color = line_colors[ ch_aa ]

            # plot the linear fit
            axs_proc_Tcomp[ aa_i - 1, 0 ].plot( kx, sigx, color=color, label=ch_aa + " linear" )
            axs_proc_Tcomp[ aa_i - 1, 1 ].plot( ky, sigy, color=color, label=ch_aa + " linear" )

        # for
        axs_proc_Tcomp[ aa_i - 1, 0 ].legend()  # turn on the legends
        axs_proc_Tcomp[ aa_i - 1, 1 ].legend()
        axs_proc_Tcomp[ aa_i - 1, 0 ].set_ylabel( 'Signal Shifts (nm)' )  # set y-axis labels

    # for

    # plot formatting
    fig_proc_Tcomp.suptitle( 'T Compensated Signal Shifts vs Curvature (X | Y)' )
    axs_proc_Tcomp[ -1, 0 ].set_xlabel( 'Curvature X (1/m)' )  # set x axis labels
    axs_proc_Tcomp[ -1, 1 ].set_xlabel( 'Curvature Y (1/m)' )

    # Plot and save plots for (T Comp.) wavelength shift vs curvature_x and curvature_y
    outfile_fig_proc_base = os.path.join( directory, outfile_base + "all-curvatures_xy-signal-shifts{}.png" )
    fig_proc.savefig( outfile_fig_proc_base.format( '' ) )  # save the processd signals shifts
    print( "Saved processed signals figure:", outfile_fig_proc_base.format( '' ) )

    fig_proc_Tcomp.savefig( outfile_fig_proc_base.format( '-T-Comp' ) )  # save the processed signals shifts
    print( "Saved T compensated processed signals figure:", outfile_fig_proc_base.format( '-T-Comp' ) )

    return total_df, proc_total_df, proc_Tcomp_total_df


# jig_process_signals

def jig_validation( directory: str, dirs_degs: dict, fbg_needle: FBGNeedle, curvatures_list: list ):
    """ Perform jig validation"""
    warnings.warn( DeprecationWarning( 'Use class implementation of jig calibration.' ) )
    # ensure that the FBG needle is up-to-date with the calibration matrices
    assert (all( aa_i in fbg_needle.cal_matrices.keys() for aa_i in range( 1, fbg_needle.num_activeAreas + 1 ) ))

    # process the FBG signals
    total_df, _, proc_Tcomp_total_df = jig_process_signals( directory, dirs_degs, fbg_needle, curvatures_list )

    # TODO: evaluate the calibration on the validation dataset


# jig_validation

def __get_argparser() -> argparse.ArgumentParser:
    """ Parse cli arguments"""
    # Setup parsed arguments
    parser = argparse.ArgumentParser( description="Perform needle jig calibration" )
    parser.add_argument( 'fbgParamFile', type=str, help='FBG parameter json file' )

    parser.add_argument( '--calib-directory', dest='calibDirectory', type=str,
                         help='Directory where all of the calibration data is.' )
    parser.add_argument( '--valid-directory', dest='validDirectory', type=str, default=None,
                         help='Directory where all of the validation data is' )

    parser.add_argument( '--angles', type=int, nargs='+', default=[ 0, 90, 180, 270 ],
                         help='Angles that will be used for the calibration' )

    parser.add_argument( '--calib-curvatures', type=float, nargs='+', default=[ 0, 0.5, 1.6, 2.0, 2.5, 3.2, 4 ],
                         help='curvatures of the calibration to be used' )
    parser.add_argument( '--valid-curvatures', type=float, nargs='+', default=[ 0, 0.25, 0.8, 1.0, 1.25, 3.125 ],
                         help='curvatures of the validation to be used' )
    parser.add_argument( '--cutoff-weight-rule', dest='cutoff_weights', type=float, nargs=3, default=None,
                         metavar=('weight<=threshold', 'threshold', 'weight>threshold'),
                         help="Weighting thresholding values for a weighted calibration procedure." )
    parser.add_argument( '--new-needleparam-file', type=str, default=None,
                         help="The output needle parameter file" )

    return parser


# __get_argparser

def main( args=None ):
    parser = __get_argparser()

    # parse the arguments
    args = parser.parse_args( args )

    # set-up
    # dir_pattern = r".*([0-9]+)_deg{0}([0-9].?[0-9]*){0}.*".format( os.sep.replace( '\\', '\\\\' ) )

    # display the arguments
    print( 'angles:', args.angles )
    print( 'needle_json:', args.fbgParamFile )
    print( 'calibDirectory:', args.calibDirectory )
    print( 'validDirectory:', args.validDirectory )
    print( 'calibCurvatures:', args.calib_curvatures )
    print( 'validCurvatures:', args.valid_curvatures )
    print( 'cutoff_weight_rule:', args.cutoff_weights )
    print()

    # load FBGNeedle
    fbg_needle = FBGNeedle.load_json( args.fbgParamFile )
    print( "Current {} Needle:".format("MCF" if isinstance(fbg_needle, MCFNeedle) else "FBG" ) )
    print( fbg_needle )
    print()

    # set-up weighting rule
    if args.cutoff_weights is not None:
        weights_rule = lambda k, i: args.cutoff_weights[ 0 ] if np.linalg.norm( k ) <= args.cutoff_weights[ 1 ] else \
            args.cutoff_weights[ 2 ]
    else:
        weights_rule = None

    # else

    # configure jig calibrator
    jig_calibrator = FBGNeedleJigCalibrator( fbg_needle, args.calibDirectory, args.calib_curvatures, args.angles,
                                             valid_directory=args.validDirectory,
                                             valid_curvatures=args.valid_curvatures, valid_angles=args.angles,
                                             weight_rule=weights_rule )

    # perform calibration and validation
    if len( jig_calibrator.valid_dataset ) == 0:  # validation dataset not configured
        print( "Performing calibration, validation not configured..." )
        jig_calibrator.process_dataset( jig_calibrator.calib_dataset, 'calibration', save=False, add_dataset=True )

        jig_calibrator.assign_dataset_type( jig_calibrator.calib_curvatures, 'calibration' )

        jig_calibrator.run_calibration( save=True, fbgneedle_param_outfile=args.new_needleparam_file )

    # if
    elif all( map( lambda k: k == 0, jig_calibrator.valid_curvatures ) ):
        print( "Performing calibration, validation not configured..." )
        jig_calibrator.process_dataset( jig_calibrator.calib_dataset, 'calibration', save=False, add_dataset=True )

        jig_calibrator.process_dataset( jig_calibrator.valid_dataset, 'validation', save=False, add_dataset=True )

        jig_calibrator.assign_dataset_type( jig_calibrator.calib_curvatures, 'calibration' )

        jig_calibrator.run_calibration( save=True, fbgneedle_param_outfile=args.new_needleparam_file )

    # elif

    else:
        print( "Performing calibration and validation..." )
        print( "Processing calibration dataset..." )
        jig_calibrator.process_dataset( jig_calibrator.calib_dataset, 'calibration', save=False, add_dataset=True )
        print()

        print( "Processing validation dataset..." )
        jig_calibrator.process_dataset( jig_calibrator.valid_dataset, 'validation', save=False, add_dataset=True )

        jig_calibrator.assign_dataset_type( jig_calibrator.calib_curvatures, 'calibration' )
        jig_calibrator.assign_dataset_type( jig_calibrator.valid_curvatures, 'validation' )

        jig_calibrator.run_calibration_validation( save=True, fbgneedle_param_outfile=args.new_needleparam_file )

    # else
    print()

    print( "Saving all data..." )
    # TODO: change processed data file names per type of calibration
    jig_calibrator.save_processed_data( outfile_base='', outdir=jig_calibrator.calib_directory )
    print( "Data saved." )
    print()

    print( "Calibrated FBG Needle:" )
    print( jig_calibrator.fbg_needle )
    print()

    print( "Program completed." )


# main

if __name__ == "__main__":
    # call main method
    main()

# if __main__

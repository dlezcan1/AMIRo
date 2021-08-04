"""
Created on Aug 3, 2020

This is a class file for FBG Needle parameterizations

@author: Dimitri Lezcano
"""
import json
from itertools import product
from os import path
from warnings import warn
from typing import Union

import fbg_signal_processing

import numpy as np


class Needle( object ):
    """ basic Needle class """

    def __init__( self, length: float ):
        """ constructor

            Args:
                - length: float, of the length of the entire needle (mm)


        """
        self._length = None

        self.length = length

    # __init__

    # =================== PROPERTIES =============================#
    @property
    def length( self ):
        return self._length

    # length

    @length.setter
    def length( self, length ):
        if not self._length and length > 0:
            self._length = length

        # if

    # length: setter

    # =============== FUNCTIONS ==================================#
    def constant_curv_2d( self, wx, ds: float = 0.5 ):
        """ returns the constant curvature shape given by the rotation about wx """
        raise NotImplementedError( "'constant_curv_2d' is not implemented yet." )
        s = np.arange( 0, self.length, ds )  # arclength points

        shape = np.zeros( (3, len( s )), dtype=float )  # 3-D shape of the needle

        rotx = lambda t: np.array( [ [ 1, 0, 0 ],
                                     [ 0, np.cos( t ), -np.sin( t ) ],
                                     [ 0, np.sin( t ), np.cos( t ) ] ] )

        return shape

    # constant_curv_2d


# class: Needle  


class FBGNeedle( object ):
    """
    This is a class for FBG Needle parameters containment.
    """

    def __init__( self, length: float, num_channels: int, sensor_location: list = [ ],
                  calibration_mats: dict = { }, weights: dict = { } ):
        """
        Constructor

        Args:
            - length: float, of the length of the entire needle (mm)
            - num_channels: int, the number of channels there are
            - sensor_location: list, the arclength locations (mm) of the AA's (default = None)
                This measurement is from the base of the needle
        """

        # data checking
        if length <= 0:
            raise ValueError( "'length' must be > 0." )

        # if

        if num_channels <= 0:
            raise ValueError( "'num_channels' must be > 0." )

        # if

        if len( sensor_location ) > 0:
            sensor_loc_invalid = [ loc > length or loc < 0 for loc in sensor_location ]
            if any( sensor_loc_invalid ):
                raise ValueError( "all sensor locations must be in [0, 'length']" )

        # if

        # property set-up (None so that they are set once)
        self._length = None
        self._num_channels = None
        self._sensor_location = None
        self._cal_matrices = { }
        self._weights = { }

        # assignments
        self.length = length
        self.num_channels = num_channels
        self.sensor_location = sensor_location
        self.cal_matrices = calibration_mats
        self.weights = weights
        self.ref_wavelengths = np.zeros( self.num_channels * self.num_activeAreas )  # reference wavelengths

    # __init__

    def __str__( self ):
        """ Magic str method """
        msg = "Needle length (mm): {}".format( self.length )
        msg += "\nNumber of FBG Channels: {:d}".format( self.num_channels )
        msg += "\nNumber of Active Areas: {:d}".format( self.num_activeAreas )
        msg += "\nSensor Locations (mm):"
        if self.num_activeAreas > 0:
            for i in range( self.num_activeAreas ):
                msg += "\n\t{:d}: {}".format( i + 1, self.sensor_location[ i ] )

            # for
        # if
        else:
            msg += " None"

        # else

        if self.cal_matrices:
            msg += "\nCalibration Matrices:"
            for loc, cal_mat in self.cal_matrices.items():
                msg += "\n\t{}: ".format( loc ) + str( cal_mat.tolist() )

                if self.weights:
                    msg += " | weight: " + str( self.weights[ loc ] )

                # if

            # for

        # if

        return msg

    # __str__

    ############################## PROPERTIES ######################################
    @property
    def length( self ):
        return self._length

    # length

    @length.setter
    def length( self, length ):
        if not self._length and length > 0:
            self._length = length

        # if

    # length: setter

    @property
    def num_aa( self ):
        DeprecationWarning( 'num_aa is deprecated. Please use num_activeAreas.' )
        return len( self.sensor_location )

    # num_aa

    @property
    def num_activeAreas( self ):
        return len( self.sensor_location )

    # num_activeAreas

    @property
    def num_channels( self ):
        return self._num_channels

    # num_channels

    @num_channels.setter
    def num_channels( self, num_chs ):
        if not self._num_channels and num_chs >= 0:
            self._num_channels = num_chs

        # if

    # num_channels: setter

    @property
    def sensor_location( self ):
        return self._sensor_location

    # sensor_locations

    @sensor_location.setter
    def sensor_location( self, sensor_locations: list ):
        if not self._sensor_location:
            if sensor_locations is None:
                self._sensor_location = None

            # if

            else:
                self._sensor_location = list( sensor_locations )

            # else 
        # if

    # sensor_location: setter

    @property
    def cal_matrices( self ):
        return self._cal_matrices

    # cal_matrices

    @cal_matrices.setter
    def cal_matrices( self, C_dict: dict ):
        for key, C in C_dict.items():
            # get the sensor location
            # check of 'AAX' format
            if isinstance( key, str ):
                loc = self.aa_loc( key )

            # if

            elif isinstance( key, int ):
                # check if it is alrady a sensor location
                if key in self.sensor_location:
                    loc = key

                # if

                # if not, check to see if it is an AA index [1, #AA]
                elif key in range( 1, self.num_activeAreas + 1 ):
                    loc = self.sensor_location[ key - 1 ]

                # elif

            # elif

            else:
                raise ValueError( "'{}' is not recognized as a valid key.".format( key ) )

            # else

            self._cal_matrices[ loc ] = C

        # for          

    # cal_matrices: setter

    @property
    def weights( self ):
        return self._weights

    # weights

    @weights.setter
    def weights( self, weights: dict ):
        for key, weight in weights.items():
            # get the sensor location
            if isinstance( key, str ):
                loc = self.aa_loc( key )

            # if

            elif isinstance( key, int ):
                # check if it is alrady a sensor location
                if key in self.sensor_location:
                    loc = key

                # if

                # if not, check to see if it is an AA index [1, #AA]
                elif key in range( 1, self.num_activeAreas + 1 ):
                    loc = self.sensor_location[ key - 1 ]

                # elif

            # elif

            else:
                raise ValueError( "'{}' is not recognized as a valid key.".format( key ) )

            # else

            # add the weight to weights
            self._weights[ loc ] = weight

        # for

    # weights setter

    ######################## FUNCTIONS ######################################

    def aa_cal( self, aax: str ):
        """ Function to get calibration matrix from AAX indexing """
        return self.cal_matrices[ self.aa_loc( aax ) ]

    # aa_cal

    def aa_idx( self, aax: str ):
        """ Function to get value from AAX indexing """
        return int( "".join( filter( str.isdigit, aax ) ) ) - 1

    # get_aa

    def aa_loc( self, aax: str ):
        """ Function to get location from AAX indexing """
        return self.sensor_location[ self.aa_idx( aax ) ]

    # aa_loc

    @staticmethod
    def assignments_aa( num_channels: int, num_active_areas: int ) -> list:
        """ Function for returning a list of AA assignments """
        return list( range( 1, num_active_areas + 1 ) ) * num_channels

    # assignments_aa

    def assignments_AA( self ) -> list:
        """ Instance method of assignments_aa """
        return FBGNeedle.assignments_aa( self.num_channels, self.num_activeAreas )

    # assignments_AA

    @staticmethod
    def assignments_ch( num_channels: int, num_active_areas: int ) -> list:
        """ Function for returning a list of CH assignments"""
        return sum( [ num_active_areas * [ ch_i ] for ch_i in range( 1, num_channels + 1 ) ], [ ] )

    # assignments_ch

    def assignments_CH( self ) -> list:
        """ Instance method of assignments_ch """
        return FBGNeedle.assignments_ch( self.num_channels, self.num_activeAreas )

    # assignments_ch

    def curvatures_raw( self, raw_signals: Union[ dict, np.ndarray ], temp_comp: bool = True ) -> \
            Union[ dict, np.ndarray ]:
        """ Determine the curvatures from signals input

                    Args:
                        raw_signals: ({AA_index: signals} | numpy array of signals (can be multi-row))
                                      must be only the raw signals
                        temp_comp: bool (Default True) of whether to use temperature compensation

                    Return:
                        (dict of {AA_index: curvature_xy} | numpy array of size 2 x num_activeAreas of curvature_xy)
        """
        curvatures = { }
        aa_assignments = self.assignments_AA()

        if isinstance( raw_signals, dict ):
            proc_signals = np.zeros( self.num_channels * self.num_activeAreas )
            raw_signals = { }
            for aa_i, raw_signal in raw_signals.items():
                # get the appropriate calibration matrix
                aa_idx = aa_i if isinstance( aa_i, int ) else int( "".join( filter( str.isdigit, aa_i ) ) )
                aa_i_mask = list( map( lambda aa: aa == aa_idx, aa_assignments ) )
                base_signal = self.ref_wavelengths[ aa_i_mask ]

                # process the signal
                proc_signal = fbg_signal_processing.process_signals( raw_signal, base_signal )
                proc_signals[ aa_i_mask ] = proc_signal

            # for

            # temperature compensate
            if temp_comp:
                proc_signals = fbg_signal_processing.temperature_compensation( proc_signals, self.num_channels,
                                                                               self.num_activeAreas )

            # if

            curvatures = self.curvatures_processed( proc_signals )

        # if

        elif isinstance( raw_signals, np.ndarray ):
            # process the signals
            proc_signals = fbg_signal_processing.process_signals( raw_signals, self.ref_wavelengths )
            if temp_comp:
                proc_signals = fbg_signal_processing.temperature_compensation( proc_signals, self.num_channels,
                                                                               self.num_activeAreas )

            # if

            if raw_signals.ndim == 1 and proc_signals.ndim == 2 and proc_signals.shape[ 0 ] == 1:
                proc_signals = proc_signals.squeeze( axis=0 )

            # if

            curvatures = self.curvatures_processed( proc_signals )

        # elif

        else:
            raise TypeError( "raw_signals must be a 'dict' or 'numpy.ndarray'" )

        # else

        return curvatures

    # curvatures_raw

    def curvatures_processed( self, proc_signals: Union[ dict, np.ndarray ] ) -> Union[ dict, np.ndarray ]:
        """ Determine the curvatures from signals input

            Args:
                proc_signals: {AA_index: processed signal} must be processed and temperature compensated

        """

        if isinstance( proc_signals, dict ):
            curvatures = { }
            for aa_i, proc_signal in proc_signals.items():
                # get the appropriate calibration matrix
                C_aa_i = self.aa_cal( f"AA{aa_i}" ) if isinstance( aa_i, int ) else self.aa_cal( aa_i )

                curvatures[ aa_i ] = C_aa_i @ proc_signal  # 2 x num_AA @ num_AA x 1

            # for

        # if

        elif isinstance( proc_signals, np.ndarray ):
            # initalize curvatures
            if proc_signals.ndim == 1:
                curvatures = np.zeros( (2, self.num_activeAreas) )
            elif proc_signals.ndim == 2:
                curvatures = np.zeros( (proc_signals.shape[ 0 ], 2, self.num_activeAreas) )
            else:
                raise IndexError( "'proc_signals' dimensions must be <= 2." )

            for aa_i in range( 1, self.num_activeAreas + 1 ):
                mask_aa_i = list( map( lambda aa: aa == aa_i, self.assignments_AA() ) )

                C_aa_i = self.aa_cal( f"AA{aa_i}" )

                if proc_signals.ndim == 1:
                    proc_signals_aa_i = proc_signals[ mask_aa_i ]
                    curvatures[ :, aa_i - 1 ] = C_aa_i @ proc_signals_aa_i

                # if

                else:
                    proc_signals_aa_i = proc_signals[ :, mask_aa_i ]
                    curvatures[ :, :, aa_i - 1 ] = proc_signals_aa_i @ C_aa_i.T

                # else
            # for

        # else

        else:
            raise TypeError( "'proc_signals' must be a 'dict' or 'numpy.ndarray'" )

        # else

        return curvatures

    # curvatures_processed

    @staticmethod
    def generate_ch_aa( num_channels: int, num_active_areas: int ) -> (list, list, list):
        """ Generate the CHX | AAY list

        """
        channels = [ f"CH{i}" for i in range( 1, num_channels + 1 ) ]
        active_areas = [ f"AA{i}" for i in range( 1, num_active_areas + 1 ) ]
        channel_active_area = [ " | ".join( (ch, aa) ) for ch, aa in product( channels, active_areas ) ]

        return channel_active_area, channels, active_areas

    # generate_ch_aa

    def generate_chaa( self ) -> (list, list, list):
        """ Instance method of generate_ch_aa"""
        return FBGNeedle.generate_ch_aa( self.num_channels, self.num_activeAreas )

    # generate_chaa

    @staticmethod
    def load_json( filename: str ):
        """ 
        This function is used to load a FBGNeedle class from a saved JSON file.
        
        Args:
            - filename: str, the input json file to be loaded.
            
        Returns:
            A FBGNeedle Class object with the loaded json files.
        
        """
        # load the data from the json file to a dict
        with open( filename, 'r' ) as json_file:
            data = json.load( json_file )

        # with

        # insert the sensor locations in order of AA
        if 'Sensor Locations' in data.keys():
            sensor_locations = [ data[ 'Sensor Locations' ][ str( key ) ] for key in
                                 sorted( data[ 'Sensor Locations' ].keys(), ) ]

        # if

        else:
            sensor_locations = None

        # else

        # insert the calibration matrices
        if "Calibration Matrices" in data.keys():
            cal_mats = { }
            for loc, c_mat in data[ "Calibration Matrices" ].items():
                if isinstance( loc, str ):
                    loc = int( "".join( filter( str.isdigit, loc ) ) )
                cal_mats[ loc ] = np.array( c_mat )

            # for

        # if

        else:
            cal_mats = { }

        # else

        if "weights" in data.keys():
            weights = { }
            for loc, weight in data[ 'weights' ].items():
                if isinstance( loc, str ):
                    loc = int( "".join( filter( str.isdigit, loc ) ) )

                weights[ loc ] = int( weight )
                # for
        # if

        else:
            weights = { }

        # else

        # instantiate the FBGNeedle class object
        fbg_needle = FBGNeedle( data[ 'length' ], data[ '# channels' ], sensor_locations,
                                cal_mats, weights )

        # return the instantiation
        return fbg_needle

    # load_json

    def save_json( self, outfile: str = "needle_params.json" ):
        """
        This function is used to save the needle parameters as a JSON file.
        
        Args:
            - outfile: str, the output json file to be saved.
        
        """
        data = { }  # initialize the json dictionary

        # place the saved data into the json file
        data[ "length" ] = self.length
        data[ "# channels" ] = self.num_channels
        data[ "# active areas" ] = self.num_activeAreas

        if self.sensor_location:
            data[ "Sensor Locations" ] = { }
            for i, l in enumerate( self.sensor_location, 1 ):
                data[ "Sensor Locations" ][ str( i ) ] = l

            # for
        # if

        if self.cal_matrices:
            data[ "Calibration Matrices" ] = { }
            for k, cal_mat in self.cal_matrices.items():
                data[ "Calibration Matrices" ][ k ] = cal_mat.tolist()

            # for
        # if

        if self.weights:
            data[ 'weights' ] = { }
            for k, weight in self.weights.items():
                data[ 'weights' ][ k ] = weight

            # for
        # if

        # write the data
        with open( outfile, 'w' ) as outfile:
            json.dump( data, outfile, indent=4 )

        # with

    # save_json

    def set_calibration_matrices( self, cal_mats: dict ):
        """ This function is to set the calibration matrices after instantiation """
        warn( DeprecationWarning( "Use function property setter obj.cal_matrices = ..." ) )
        self.cal_matrices = cal_mats

    # set_calibration_matrices

    def set_weights( self, weights: dict ):
        """ This function is to set the weighting of their measurements """

        self.weights = weights

    # set_weights


# class: FBGNeedle


def main( args=None ):
    # create and FBGNeedle json
    directory = "../data/needle_3CH_4AA_v3"

    # FBG needle parameters
    length = 160  # mm
    num_chs = 3
    aa_locs_tip = np.cumsum( [ 11, 20, 35, 35 ] )[ ::1 ]  # 4 AA needle
    aa_locs = (length - aa_locs_tip).tolist()

    # new FBG needle
    new_needle = FBGNeedle( length, num_chs, aa_locs )

    print( "New needle parameters:" )
    print( new_needle )
    print()

    save_file = path.join( directory, 'needle_params.json' )
    new_needle.save_json( save_file )
    print( f"Saved new needle json: {save_file}" )


# main

# for debugging purposes and creating new FBGneedle param files
if __name__ == "__main__" or False:
    # TODO: argparsing
    main()

# if: __main__

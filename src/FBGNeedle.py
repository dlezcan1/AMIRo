'''
Created on Aug 3, 2020

This is a class file for FBG Needle parameterizations

@author: Dimitri Lezcano
'''
import json
import numpy as np


class FBGNeedle( object ):
    '''
    This is a class for FBG Needle parameters containment.
    '''

    def __init__( self, length: float, num_channels: int, sensor_location: list = [],
                  calibration_mats: dict = None ):
        '''
        Constructor
        
        Args:
            - length: float, of the length of the entire needle (mm)
            - num_channels: int, the number of channels there are
            - sensor_location: list, the arclength locations (mm) of the AA's (default = None)
                This measurement is from the base of the needle
        '''

        # data checking
        if length <= 0:
            raise ValueError( "'length' must be > 0." )
        
        # if
        
        if num_channels <= 0:
            raise ValueError( "'num_channels' must be > 0." )
        
        # if
        if len( sensor_location ) > 0:
            sensor_loc_invalid = [loc > length or loc < 0 for loc in sensor_location]
            if any( sensor_loc_invalid ):
                raise ValueError( "all sensor locations must be in [0, 'length']" )
            
            # if
            if calibration_mats:
                if not set( calibration_mats.keys() ).issubset( set( sensor_location ) ):
                    raise IndexError( "There is a mismatch in the calibration matrices to sensor locations." )
                    
                # if
            # if
        # if
        
        # property set-up (None so that they are set once)
        self._length = None
        self._num_channels = None
        self._sensor_location = None
        
        # assignments
        self.length = length
        self.num_channels = num_channels
        self.sensor_location = sensor_location        
        self.cal_matrices = calibration_mats
        
    # __init__
    
    def __str__( self ):
        """ Magic str method """
        msg = "Needle length (mm): {}".format( self.length )
        msg += "\nNumber of FBG Channels: {:d}".format( self.num_channels )
        msg += "\nNumber of Active Areas: {:d}".format( self.num_aa )
        msg += "\nSensor Locations (mm):"
        if self.num_aa > 0:
            for i in range( self.num_aa ):
                msg += "\n\t{:d}: {}".format( i + 1, self.sensor_location[i] )
                
            # for
        # if
        else:
            msg += " None"
            
        # else
        
        if self.cal_matrices:
            msg += "\nCalibration Matrices:"
            for loc, cal_mat in self.cal_matrices.items():
                msg += "\n\t{}: ".format( loc ) + str( cal_mat.tolist() )
            
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
        return len( self.sensor_location )
 
    # num_aa
    
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
    def sensor_location( self, sensor_locations ):
        if not self._sensor_location:
            if sensor_locations is None:
                self._sensor_location = None
                
            # if
            
            else:
                self._sensor_location = sorted( list( set( sensor_locations ) ), reverse = True )  # remove duplicates
            
            # else 
        # if
    # sensor_location: setter
    
############################## FUNCTIONS ######################################
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
            sensor_locations = [data['Sensor Locations'][str( key )] for key in sorted( data['Sensor Locations'].keys() )]
        
        # if
        
        else:
            sensor_locations = None
        
        # else
            
        # insert the calibration matrices
        if "Calibration Matrices" in data.keys():
            cal_mats = {}
            for loc, c_mat in data["Calibration Matrices"].items():
                cal_mats[int( loc )] = np.array( c_mat )
                
            # for
        
        # if
        
        else:
            cal_mats = None
            
        # else
        
        # instantiate the FBGNeedle class object
        fbg_needle = FBGNeedle( data['length'], data['# channels'], sensor_locations, cal_mats )
        
        # return the instantiation
        return fbg_needle
    
    # load_json
    
    def save_json( self, outfile: str = "needle_params.json" ):
        """
        This function is used to save the needle parameters as a JSON file.
        
        Args:
            - outfile: str, the output json file to be saved.
        
        """
        data = {}  # initialize the json dictionary
        
        # place the saved data into the json file
        data["length"] = self.length
        data["# channels"] = self.num_channels
        data["# active areas"] = self.num_aa
        
        if self.sensor_location:
            data["Sensor Locations"] = {}
            for i, l in enumerate( self.sensor_location, 1 ):
                data["Sensor Locations"][ str( i ) ] = l
                
            # for
        # if
        
        if self.cal_matrices:
            data["Calibration Matrices"] = {}
            for k, cal_mat in self.cal_matrices.items():
                data["Calibration Matrices"][k] = cal_mat.tolist()
                
            # for
            
        # if
        
        # write the data
        with open( outfile, 'w' ) as outfile:
            json.dump( data, outfile, indent = 4 )
            
        # with
        
    # save_json
    
    def set_calibration_matrices( self, cal_mats: dict ):
        """ This function is to set the calibration matrices after instantiation """
        # data checking
        if not set( cal_mats.keys() ).issubset( set( self.sensor_location ) ):
            raise IndexError( "There is a mismatch in the calibration matrices to sensor locations." )
            
        # if
        
        self.cal_matrices = cal_mats
        
    # set_calibration_matrices
    
# class: FBGNeedle


# for debugging purposes
if __name__ == "__main__" or False:
    
    # directory to save in
    directory = "../FBG_Needle_Calibration_Data/needle_3CH_4AA/"
    save_bool = True
    
    # needle parameters
    length = 200  # mm
    num_chs = 3
    aa_locs_tip = np.cumsum( [10, 20, 35, 35] ) 
    aa_locs = sorted( ( 200 - aa_locs_tip ).tolist(), reverse = True )
    cal_mats = None
    
    # create and save the new 
    test = FBGNeedle( length, num_chs, aa_locs )
    print( str( test ) )
    
    if save_bool:
        save_file = directory + "needle_params.json"
        test.save_json( save_file )
        print( "Saved file: " + save_file )
        
        test2 = FBGNeedle.load_json( save_file )
        print( "after load" )
        print( str( test2 ) )
        
    # if
    
# if: __main__


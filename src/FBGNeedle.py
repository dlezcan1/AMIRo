'''
Created on Aug 3, 2020

This is a class file for FBG Needle parameterizations

@author: Dimitri Lezcano
'''
import json


class FBGNeedle( object ):
    '''
    This is a class file for FBG Needle parameters
    '''

    def __init__( self, length: float, num_channels: int, sensor_location: list = None ):
        '''
        Constructor
        
        Args:
            - length: float, of the length of the entire needle (mm)
            - num_channels: int, the number of channels there are
            - sensor_location: list, the arclength locations (mm) of the AA's (default = None)
        '''
        
        # data checking
        if length <= 0:
            raise ValueError( "'length' must be > 0." )
        
        if num_channels <= 0:
            raise ValueError( "'num_channels' must be > 0." )
        
        sensor_loc_valid = [loc > length or loc < 0 for loc in sensor_location]
        if any( sensor_loc_valid ):
            raise ValueError( "all sensor locations must be in [0, 'length']" )
        
        # assignments
        self.length = length
        self.num_channels = num_channels
        self.num_aa = len( sensor_location )
        self.sensor_location = sorted( sensor_location )
        
    # __init__
    
    def save_json( self, filename: str = "needle_params.json", directory: str = "" ):
        """
        This function is used to save the needle parameters as a JSON file.
        
        Args:
            - filename: str, the output json file to be saved.
        
        """
        data = {}  # initialize the json dictionary
        
        # place the saved data into the json file
        data["length"] = self.length
        data["# channels"] = self.num_channels
        data["# active areas"] = self.num_aa
        data["Sensor Locations"] = {}
        for i, l in enumerate( self.sensor_location, 1 ):
            data["Sensor Locations"][ str( i ) ] = l
            
        # for
        
        # write the data
        with open( directory + filename, 'w' ) as outfile:
            json.dump( data, outfile, indent = 4 )
            
        # with
        
    # save_json
    
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
        sensor_locations = [data['Sensor Locations'][str( key )] for key in sorted( data['Sensor Locations'].keys() )]
        
        # instantiate the FBGNeedle class object
        fbg_needle = FBGNeedle( data['length'], data['# channels'], sensor_locations )
        
        # return the instantiation
        return fbg_needle
    
    # load_json
    
    def __str__( self ):
        """ Magis str method """
        msg = "Needle length (mm): {:f}".format( self.length )
        msg += "\nNumber of FBG Channels: {:d}".format( self.num_channels )
        msg += "\nNumber of Active Areas: {:d}".format( self.num_aa )
        msg += "\nSensor Locations (mm):"
        for i in range( self.num_aa ):
            msg += "\n\t{:d}: {:f}".format( i + 1, self.sensor_location[i] )
            
        # for
        
        return msg
        
    # __str__
        
# class: FBGNeedle


# for debugging purposes
if __name__ == "__main__" and False:

    directory = "C:/Users/dlezcan1/Desktop/"
    
    print( "before" )
    test = FBGNeedle( 30, 90, [0, 20, 30] )
    print( str( test ) )
    
    save_file = directory + "test.json"
    test.save_json( save_file )
    print( "Saved file: " + save_file )
    
    test2 = FBGNeedle.load_json( save_file )
    print( "after load" )
    print( str( test2 ) )


'''
Created on Jan 2, 2020

@author: Dimitri Lezcano

@summary: Class in order to fit a B-Spline to data smoothly
'''
import numpy as np
import scipy.interpolate as interp
from scipy.integrate import quad
from warnings import warn


class BSpline1D():
    '''
    BSpline 1-D
    '''

    def __init__( self, x: np.ndarray, y: np.ndarray, k: int = 3, xb: float = None,
                  xe: float = None, w: np.ndarray = None ):
        '''
        Returns an interpolator that uses 0 knots and instead scales the values
        between 0 and 1
        '''
        self.__x = x
        self.__y = y
        self.__qmin = x.min()
        self.__qmax = x.max()
        self.__tck = interp.splrep( self._scale( x ), y,
                                  w = w, xb = xb, xe = xe, k = k, t = [0.25, .5, 0.75] )
        
    # __init__

#================= PROPERTIES ======================= 
    @property
    def qmin( self ):
        return self.__qmin
    
    # property getter: qmin
    
    @qmin.setter
    def qmin( self, qmin ):        if not self.__qmin:
            self.__qmin = qmin
            
        # if
        
        else:
            raise ValueError( "qmin already set." )
        
        # else
        
    # property setter: qmin
    
    @property
    def qmax( self ):
        return self.__qmax
    
    # property getter: qmax
    
    @qmax.setter
    def qmax( self, qmax ):
        if not self.__qmax:
            self.__qmax = qmax
            
        # if
        
        else:
            raise ValueError( "qmax already set." )
        
        # else
        
    # property setter: qmin    
    
    @property
    def tck( self ):
        return self.__tck
    
    # property getter: tck
    
    @tck.setter
    def tck( self, tck ):
        if not self.__tck:
            self.__tck = tck
            
        # if
        
        else:
            raise ValueError( "x already set." )
        
        # else
        
    # property setter: tck 
    
    @property
    def x( self ):
        return self.__x
    
    # property getter: x
    
    @x.setter
    def x( self, x ):
        if not self.__x:
            self.__x = x
            
        # if
        
        else:
            raise ValueError( "x already set." )
        
        # else
        
    # property setter: x   
    
    @property
    def y( self ):
        return self.__y
    
    # property getter: y
    
    @y.setter
    def y( self, y ):
        if not self.__y:
            self.__y = y
            
        # if
        
        else:
            raise ValueError( "y already set." )
        
        # else
        
    # property setter: x  
    
#================= METHODS ======================= 

    def __call__( self, x , der: int = 0 ):
        """ Return the function's output for a specific derivative """
        
        return interp.splev( self._scale( x ), self.tck, der = der, ext = 0 )
    
    # __call__
    
    def _scale( self, data: np.ndarray ):
        """ Scale the input to 0 and 1 """
        if self.qmin == self.qmax:
            retval = np.ones( data.shape )
        
        # if
        
        else:
            retval = ( data - self.qmin ) / ( self.qmax - self.qmin )
            
        # else
        
        if np.any( retval > 1 ) or np.any( retval < 0 ):
            warn( f"Scaling data is out of the range of the bounding box: [{self.qmin},{self.qmax}]" )
        
        # if
        
        return retval
    
    # _scale
    
    def arclength( self, x1: float, x2:float ):
        integrand = lambda x: np.sqrt( 1 + self.__call__( x, der = 1 ) ** 2 )
        
        return quad( integrand, x1, x2 )[0]
    
    # arclength
    
    def eval_unscale ( self, x: np.ndarray, der: int = 0 ):
        """Evaluated the spline without scaling """
        return interp.splev( x, self.tck, der = der, ext = 0 )
    
    # eval_unscale
    
    def deriv_x ( self, x: np.ndarray, der: int = 0 ):
        
        if self.qmax == self.qmin:
            retval = 0
            
        else:
            retval = 1 / ( self.qmax - self.qmin )
        
        return retval
    
    # deriv_x
    
    def integrate( self, a: float, b: float, der: int = 0 ):
        """ Not implemented """
        raise NotImplementedError( 'integrate is not impleemented' )
    
    # integrate
    
    def unscale( self, scaled_data: np.ndarray ):
        """ Returns the unscaled data """
        if self.qmin == self.qmax:
            retval = self.qmax * np.ones( scaled_data.shape )
        
        # if
        else:
            retval = ( self.qmax - self.qmin ) * scaled_data + self.qmin
            
        # else
        
        return retval
    
    # unscale

# BSpline1D

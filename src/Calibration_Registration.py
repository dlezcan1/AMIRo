'''
Created on Oct 17, 2019

@author: Dimitri Lezcano and Hyunwoo Song

@name: Calibration_Registration

@summary: This module is to provide python functions of calibration and 
          registration for computer-integrated surgical interventions.
'''

import transforms3d_extend
import numpy as np
from scipy.interpolate import BPoly


def correctDistortion( c: np.ndarray, vector: np.ndarray , qmin, qmax ):
    """This function is to perform the 3-D Bernstein polynomial function
       represented in the tensor form of the "Interpolation" slide deck
                   
       @author Dimitri Lezcano
       
       @param c:      the coefficient matrix that must be a column vector
       
       @param vector: the 3-D vector to be evaluated
       
       @param qmin:   the minimum value for scaling
       
       @param qmax:   the maximum value for scaling
       
       @return: returns a 3-D vector as calculated by the polynomial.
       
    """
    # x, y, z = scale_to_box( vector, qmin, qmax )[0]
    scaled_vec = scale_to_box( vector, qmin, qmax )[0]
    
    #x = scaled_vec[0]
    #y = scaled_vec[1]
    #z = scaled_vec[2]
    #print("x,y,z : ", x, y, z)
    #assert ( x <= 1 and x >= 0 )
    #assert ( y <= 1 and y >= 0 )
    #assert ( z <= 1 and z >= 0 )
    
    N = int( np.ceil( ( len( c ) ) ** ( 1 / 3 ) ) ) - 1
    
    bern_Matrix = generate_berntensor( vector, qmin, qmax, N )
    
    retval = ( bern_Matrix.dot( c ) ).reshape( -1 )
    
    return retval

# correctDistortion


def generate_berntensor( X: np.ndarray, qmin: float, qmax: float, order: int ):
    """Function to generatea tensor of the 3-D Bernstein functions where 
       F_ijk = b_i(x)*b_j(y)*b_k(z).
    
       @author: Dimitri Lezcano
       
       @param X:     the input to be used for the bernstein tensor
       
       @param qmin:  the minimum value for scaling
       
       @param qmax:  the maximum value for scaling
       
       @param order: the order of which you would like the 
                     the Bernstein polynomials to be.
                     
       @return: a numpy array of the Bernstein tensor
       
    """
    bern_basis = generate_Bpoly_basis( order )
    
    X_prime = scale_to_box( X, qmin, qmax )[0]
    if X.ndim > 1:
        X_px = X_prime[:, 0].reshape( ( -1, 1 ) )
        X_py = X_prime[:, 1].reshape( ( -1, 1 ) )
        X_pz = X_prime[:, 2].reshape( ( -1, 1 ) )
        bern_matrix = np.zeros( ( len( X ), ( order + 1 ) ** 3 ) )
    
    # if
    
    else:
        X_px, X_py, X_pz = X_prime
        bern_matrix = np.zeros( ( 1, ( order + 1 ) ** 3 ) )
    
    # else
    
    bern_ijk = lambda i, j, k: ( ( bern_basis[i]( X_px ) ) * ( bern_basis[j]( X_py ) ) * 
                                 ( bern_basis[k]( X_pz ) ) )
    
    for i in range( order + 1 ):
        for j in range( order + 1 ):
            for k in range( order + 1 ):
                val = bern_ijk( i, j, k )
                val = val.reshape( -1 )
                bern_matrix[:, i * ( order + 1 ) ** 2 + j * ( order + 1 ) + k] = val
                
            # for
        # for
    # for
    
    #assert ( np.min( bern_matrix ) >= 0 )
    #assert ( np.max( bern_matrix ) <= 1 )
    
    return bern_matrix
    
# generate_berntensor


def point_cloud_reg_SVD( a, b ):
    """ This function read the two coordinate systems and
        calculate the point-to-point registration function.
        This algorithm is explained in class and implemented as taught.
        The relationship between a, b, and the result ("F") is
            b = F a
        This function will use an SVD in order to determine the corresponding
        frame transformation from quaternions

        @author: Dimitri Lezcano

        @param a: the input, numpy array where the vectors are the rows of the
                  matrix
                  
        @param b: the corresponding output, numpy array where the vectors are
                  the rows of the matrix

        @return: F which is a dictionary consist of 'Ratation' as a rotation
                matrix and 'Trans' as a translational vector
    
    """
    mean_a = np.mean( a, axis = 0 )
    mean_b = np.mean( b, axis = 0 )
    
    # Compute for mean and subtract from a, b, respectively
    a_hat = a - mean_a
    b_hat = b - mean_b
    
    M = np.empty( ( 0, 4 ) )
    for ai, bi in zip( a_hat, b_hat ):
        top_left = 0
        top_right = bi - ai
        bottom_left = top_right.reshape( ( -1, 1 ) )
        bottom_right = transforms3d_extend.skew( top_right )
        top = np.append( top_left, top_right )
        bottom = np.hstack( ( bottom_left, bottom_right ) )
        Mi = np.vstack( ( top, bottom ) )
        M = np.append( M, Mi, axis = 0 )
    
    # for
    
    u, s, v = np.linalg.svd( M )
    
    q = v[:, 3]
    
    R = transforms3d_extend.quaternions.quat2mat( q )
    
    p = mean_b - R.dot( mean_a )
    
    F = {'Rotation': R, 'Trans': p}
    
    return F
    
# point_cloud_reg_SVD


def point_cloud_reg_Arun( a, b ):
    """ This function read the two coordinate systems and
        calculate the point-to-point registration function.
        This algorithm is explained in class and implemented as taught.
        The relationship between a, b, and the result ("F") is
            b = F a
        This function will use an Arun's method to determine the Rotation matrix.

        @author: Dimitri Lezcano

        @param a: the input, numpy array where the vectors are the rows of the
                  matrix
                  
        @param b: the corresponding output, numpy array where the vectors are
                  the rows of the matrix

        @return: F which is a dictionary consist of 'Ratation' as a rotation
                matrix and 'Trans' as a translational vector
    
    """
    mean_a = np.mean( a, axis = 0 )
    mean_b = np.mean( b, axis = 0 )
    
    # Compute for mean and subtract from a, b, respectively
    a_hat = a - mean_a
    b_hat = b - mean_b

    # Compute for H
    mult = np.multiply( a_hat, b_hat )
    ab_xx = np.sum( mult[:, 0] )
    ab_yy = np.sum( mult[:, 1] )    
    ab_zz = np.sum( mult[:, 2] )

    ab_xy = np.sum( np.multiply( a_hat[:, 0], b_hat[:, 1] ) )
    ab_xz = np.sum( np.multiply( a_hat[:, 0], b_hat[:, 2] ) )
    ab_yx = np.sum( np.multiply( a_hat[:, 1], b_hat[:, 0] ) )
    ab_yz = np.sum( np.multiply( a_hat[:, 1], b_hat[:, 2] ) )
    ab_zx = np.sum( np.multiply( a_hat[:, 2], b_hat[:, 0] ) )
    ab_zy = np.sum( np.multiply( a_hat[:, 2], b_hat[:, 1] ) )
    
    H = np.array( [[ab_xx, ab_xy, ab_xz], [ab_yx, ab_yy, ab_yz],
                   [ab_zx, ab_zy, ab_zz]] )
    
    u, s, v = np.linalg.svd( H )
    
    R = v.dot( u.T )
    if np.linalg.det( R ) < 0: 
        R[:, 3] *= -1
        
    p = mean_b - R.dot( mean_a )
    
    F = {'Rotation': R, 'Trans': p}
    
    return F
    
# ponit_cloud_reg_Arun


def point_cloud_reg( a, b ):
    """ This function read the two coordinate systems and
        calculate the point-to-point registration function.
        This algorithm is explained in class and implemented as taught.
        The relationship between a, b, and the result ("F") is
            b = F a
        Basically, this function compute the H matrix and calculate the
        unit quaternion at the largest eigen value of H.
        Then it uses quaternion to calculate the rotation matrix.
        Translation vector is calculated by 
            trans = b - R dot a 

        @author: Hyunwoo Song

        @param a: the input, numpy array where the vectors are the rows of the
                  matrix
                  
        @param b: the corresponding output, numpy array where the vectors are
                  the rows of the matrix

        @return: F which is a dictionary consist of 'Ratation' as a rotation
                matrix and 'Trans' as a translational vector
    
    """
    mean_a = np.mean( a, axis = 0 )
    mean_b = np.mean( b, axis = 0 )
    
    # Compute for mean and subtract from a, b, respectively
    a_hat = a - mean_a
    b_hat = b - mean_b

    # Compute for H
    mult = np.multiply( a_hat, b_hat )
    ab_xx = np.sum( mult[:, 0] )
    ab_yy = np.sum( mult[:, 1] )    
    ab_zz = np.sum( mult[:, 2] )

    ab_xy = np.sum( np.multiply( a_hat[:, 0], b_hat[:, 1] ) )
    ab_xz = np.sum( np.multiply( a_hat[:, 0], b_hat[:, 2] ) )
    ab_yx = np.sum( np.multiply( a_hat[:, 1], b_hat[:, 0] ) )
    ab_yz = np.sum( np.multiply( a_hat[:, 1], b_hat[:, 2] ) )
    ab_zx = np.sum( np.multiply( a_hat[:, 2], b_hat[:, 0] ) )
    ab_zy = np.sum( np.multiply( a_hat[:, 2], b_hat[:, 1] ) )
    
    H = np.array( [[ab_xx, ab_xy, ab_xz], [ab_yx, ab_yy, ab_yz],
                   [ab_zx, ab_zy, ab_zz]] )

    # Compute G
    H_trace = np.trace( H )
    Delta_trans = np.array( [H[1, 2] - H[2, 1], H[2, 0] - H[0, 2], H[0, 1] - H[1, 0]] )
    Delta = Delta_trans.reshape( ( -1, 1 ) )

    G = np.vstack( ( np.hstack( ( H_trace, Delta_trans ) ),
                      np.hstack( ( Delta, H + H.T - H_trace * np.eye( 3 ) ) ) ) )
    
    a_eigenVal, m_eigenVec = np.linalg.eig( G )
    
    # unit quaternion
    q = m_eigenVec[:, np.argmax( a_eigenVal )]
    
    # Calculate R using unit quaternion
    R_00 = q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2
    R_01 = 2 * ( q[1] * q[2] - q[0] * q[3] )
    R_02 = 2 * ( q[1] * q[3] + q[0] * q[2] )
    R_10 = 2 * ( q[1] * q[2] + q[0] * q[3] )
    R_11 = q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2
    R_12 = 2 * ( q[2] * q[3] - q[0] * q[1] )
    R_20 = 2 * ( q[1] * q[3] - q[0] * q[2] )
    R_21 = 2 * ( q[2] * q[3] + q[0] * q[1] )
    R_22 = q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2

    R = np.array( [[R_00, R_01, R_02], [R_10, R_11, R_12], [R_20, R_21, R_22]] )
    R_pack = transforms3d_extend.quaternions.quat2mat( q )

        # Calculate translation

    p = mean_b - R.dot( mean_a ) 
    
    F = {'Rotation': R, 'Trans': p}
    return F

# point_cloud_reg


def pointer_calibration( transformation_list: list ):
    """Function that determines the parameters of the pointer given a pivot 
    calibration data for the pointer. 
    This method is implemented as in class. 
    The least squares function used here is proved by 'numpy.linalg.lstsq'.
    
    Solves the least squares problem of:
      ...       ...                ...
    { R_j      -I  }{ p_ptr  } = { -p_j }
      ...      ...    p_post       ...
    where:
    -> (R_j,p_j) is the j-th transformation of the pivot
    -> p_ptr     is the vector of the pointer posistion relative to the tracker
    -> p_post    is the vector position of the post. 
    
    @author: Dimitri Lezcano
      
    @param transformation_list: A list of the different transformation matrices 
                                from the pivot calibration
    
    @return: [p_ptr, p_post] where they are both 3-D vectors. 
    """
    
    coeffs = np.array( [] )
    translations = np.array( [] )
    for i, transform in enumerate( transformation_list ):
        # split the transformations into their base matrices and vectors
        # zoom and shear assumed to be ones and zeros, respectively
#         print("transform: \n", transform)
        p, R, _, _ = transforms3d_extend.affines.decompose44( transform ) 
        #R = transform[:3, :3]
        #p = transform[:3, 3]
#         print("R: \n", R)
#         print("p: \n", p)
        C = np.hstack( ( R, -np.eye( 3 ) ) )
        if i == 0:  # instantiate the sections
            coeffs = C
            translations = -p
        
        # if
        
        else:  # add to the list
            coeffs = np.vstack( ( coeffs, C ) )
            translations = np.append(translations, -p)
            
        # else
    # for
    #print("transformation_list: \n", transformation_list)
    #print("coeffs: \n", coeffs)
    #print("translations: \n", translations)
        
    lst_sqr_soln, resid, rnk, sng = np.linalg.lstsq( coeffs, translations, None )
    # p_ptr  is indexed 0-2
    # p_post is indexed 3-5
#     print("calibration")
    
    return [lst_sqr_soln[:3], lst_sqr_soln[3:]]

# pointer_calibration


def generate_Bpoly_basis( N: int ):
    """This function is to generate a basis of the bernstein polynomial basis
    
       @author: Dimitri Lezcano
       
       @param N: an integer representing the highest order or the 
                 Bernstein polynomial.
    
       @return:  A list of Bernstein polynomial objects of size N, that will 
                 individually be B_0,n, B_1,n, ..., B_n,n
                
    """
    zeros = np.zeros( N + 1 )
    
    x_break = [0, 1]
    basis = []
    
    for i in range( N + 1 ):
        c = np.copy( zeros )
        c[i] = 1
        c = c.reshape( ( -1, 1 ) )
        basis.append( BPoly( c, x_break ) )
        
    # for 
    
    return basis

# generate_Bpoly


def scale_to_box( X: np.ndarray , qmin, qmax ):
    """A Function to scale an input array of vectors and return 
       the scaled version from 0 to 1.
       
       @author Dimitri Lezcano
       
       @parap X:    a numpy array where the rows are the corresponding vectors
                    to be scaled.
                 
       @param qmin: a value that represents the minimum value for scaling
       
       @param qmax: a value that represents the maximum value for scaling
     
       @return: X', the scaled vectors given from the function.
       
    """
    div = np.linalg.norm( qmax - qmin )
    
    X_prime = ( X - qmin ) / div  # normalized input
    
    return X_prime, qmin, qmax
    
# scale_to_box


def undistort( X: np.ndarray, Y :np.ndarray, order:int, qmin = None, qmax = None ):
    """Function to undistort a calibration data set using Bernstein polynomials.
       Implemented for 3-D case only.
    
       Solving the least squares problem of type:
          ...        ...        ...       c_0       ...
       ( B_0,N(x_j)  ...    B_N,N(x_j) )( ... ) = ( p_j )
          ...        ...        ...       c_N       ...
          
       @author: Dimitri Lezcano
       
       @param X: The input parameters to be fit to Y
       
       @param Y: The output parameters to be fit from X
       
       @param order: The highest order that would like to be fitted of the 
                     Bernstein polynomial
                     
       @param qmin (optional):   a floating point number representing the min
                                 value for scaling
     
       @param qmax (optional):   a floating point number representing the min
                                 value for scaling
       
       @return: A Bernstein Polynomial object with the fitted coefficients
    
    """
    if isinstance( qmin, type( None ) ):
        qmin = np.min( X )
        
    if isinstance( qmax, type( None ) ):
        qmax = np.max( X )
   
    bern_Matrix = generate_berntensor( X, qmin, qmax, order )
    
    lstsq_soln, _, _, _ = np.linalg.lstsq( bern_Matrix, Y, None )
    return lstsq_soln, qmin, qmax
    
# undistort

import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys, os
from scipy.interpolate import splrep, splev, CubicSpline
from scipy.integrate import quad
from scipy.optimize import fsolve, leastsq
from matplotlib.pyplot import draw
import matplotlib.pyplot as plt
# import scipy


def load_image( filename ):
	img = cv2.imread( filename, cv2.IMREAD_COLOR )
	gray_image = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

	return img, gray_image


def set_ROI( image, crop_area ):
	''' Note that origin of images is typically top lefthand corner
	crop = (leftX buffer, rightX buffer, topY buffer, bottomY buffer)'''
	startX = crop_area[0]
	endX = image.shape[1] - crop_area[1]
	startY = crop_area[2]
	endY = image.shape[0] - crop_area[3]

	cropped = image[startY:endY, startX:endX]

	return cropped

# set_ROI


def set_ROI_box( image, crop_area ):
	""" Sets a box around the region of interest and returns what is
		in the box
		
		crop = (top_leftX, top_leftY, bottom_rightX, bottom_rightY)
	"""
	tlx, tly, brx, bry = crop_area

	cropped_image = image[tly:bry, tlx:brx]
	
	return cropped_image

# set_ROI_box


def binary( image ):
	thresh_dark = 210
	thresh_light = 230
	binary_img = np.copy( image )
	binary_img[binary_img <= thresh_dark] = 0
	# binary_img[binary_img > thresh_light] = 0
	binary_img[binary_img != 0] = 255

	bor1 = [300, 800, 0, 65]  # xleft, xright, ytop, ybottom for the top, blackout
	binary_img[bor1[2]:bor1[3], bor1[0]:bor1[1]] = 0

	skeleton = np.copy( binary_img ) / 255
	skeleton = skeletonize( skeleton )
	binary_img = skeleton.astype( np.uint8 ) * 255

	# cv2.imshow('binary', binary_img)
	# cv2.waitKey(0)

	return binary_img


def gen_kernel( shape ):
	"""Function to generate the shape of the kernel for image processing

	@param shape: 2-tuple (a,b) of integers for the shape of the kernel
	@return: returns axb numpy array of value of 1's of type uint8
	"""
	return np.ones( shape, np.uint8 )


def canny_edge_detection( image ):
	thresh1 = 25
	thresh2 = 225
	bor1 = [300, 800, 0, 65]  # xleft, xright, ytop, ybottom for the top, blackout
	bor2 = [700, 930, 90, image.shape[0]]  # xleft, xright, ytop, ybottom for the bottom, blackout

	img = np.copy( image )

	# edges = cv2.Canny(image, thresh1, thresh2)

	# # Canny Filtering for Edge detection
	canny1 = cv2.Canny( img, thresh1, thresh2 )
	# cv2.imshow('canny1 before',canny1)

	# # Remove (pre-determined for simplicity in this code) artifacts manually
	# # I plan to make this part of the algorithm to be incorproated into GUI
	canny1[bor1[2]:bor1[3], bor1[0]:bor1[1]] = 0
	canny1[bor2[2]:bor2[3], bor2[0]:bor2[1]] = 0
	# cv2.imshow('canny1 after',canny1)
	# cv2.waitKey(0)

	# worked for black background
	kernel = gen_kernel( ( 7, 7 ) )
	canny1_fixed = cv2.morphologyEx( canny1, cv2.MORPH_CLOSE, kernel )
	# cv2.imshow('canny1 morph_close',canny1_fixed)

	kernel = gen_kernel( ( 9, 9 ) )
	canny1_fixed = cv2.dilate( canny1_fixed, kernel, iterations = 2 )
	# cv2.imshow('canny1 dilate',canny1_fixed)

	kernel = gen_kernel( ( 11, 31 ) )
	canny1_fixed = cv2.erode( canny1_fixed, kernel, iterations = 1 )
	# cv2.imshow('canny1 erode',canny1_fixed)

	kernel = gen_kernel( ( 7, 7 ) )
	canny1_fixed = cv2.morphologyEx( canny1_fixed, cv2.MORPH_OPEN, kernel )
	# cv2.imshow('canny1 morph_open',canny1_fixed)

	canny1_fixed = cv2.erode( canny1_fixed, kernel, iterations = 1 )
	# cv2.imshow('canny1 erode2',canny1_fixed)
	# cv2.waitKey(0)

	retval = canny1_fixed

	return retval


def get_centerline( edge_image ):
	# centerline_img = np.zeros(edge_image.shape)

	# for col in range(edge_image.shape[1]):
	# 	nonzero = np.argwhere(edge_image[:,col] != 0)

	# 	if len(nonzero) != 0:
	# 		center_row = int(np.rint(np.mean(nonzero)))
	# 		centerline_img[center_row, col] = 255

	# try skeletonize
	binary = np.copy( edge_image ) / 255
	skeleton = skeletonize( binary )
	skeleton = skeleton.astype( np.uint8 ) * 255

	return skeleton


def stitch( canny_img, binary_img ):
	binary_nonzero = np.argwhere( binary_img )  # find all nonzero indices

	# sort by x-coordinate
	binary_nonzero = sorted( binary_nonzero, key = lambda element: element[1] )
	binary_nonzero = np.asarray( binary_nonzero )  # convert back to numpy array

	canny_nonzero = np.argwhere( canny_img )  # find all nonzero indices
	canny_max_idx = np.argmax( canny_nonzero[:, 1] )  # find the index of rightmost point
	canny_max = canny_nonzero[canny_max_idx][1]  # find the index in original image

	stitch_img = np.copy( canny_img )
	# find where rightmost point stops in binary image
	binary_start = np.argwhere( binary_nonzero[:, 1] == canny_max )[0][0]
	
	# add binary_img points to stitch_img
	for b in range( binary_start, binary_nonzero.shape[0] ):
		coord = binary_nonzero[b]
		stitch_img[coord[0], coord[1]] = 255

	return stitch_img

# stitch


def find_param_along_poly ( poly: np.poly1d, x0: float, target_length: float ):
	deriv_1 = np.polyder( poly, 1 )
	integrand = lambda x: np.sqrt( 1 + ( deriv_1( x ) ) ** 2 )
	
	arc_length = lambda x: quad( integrand, x0, x )[0] 
	cost_fn = lambda x: np.abs( target_length - arc_length( x ) )
	
	ret_x = fsolve( cost_fn, x0 )[0]
	err = target_length - arc_length( ret_x )
	
	return ret_x, err
	
# find_param_along_poly


def find_param_along_spline ( s, x0: float, target_length: float ):
	costfn = lambda x: np.abs( target_length - arclength_spline( s, x0, x ) )
	ret_x = fsolve ( costfn, x0 ) [0]
	
	err = target_length - arclength_spline( s, x0, ret_x )

	return ret_x, err

# find_param_along_spline


def arclength( poly: np.poly1d, a: float, b: float ):
	deriv_1 = np.polyder( poly, 1 )
	integrand = lambda x: np.sqrt( 1 + ( deriv_1( x ) ) ** 2 )
	
	return quad( integrand, a, b )[0]

# arclength


def arclength_spline ( s, a: float, b: float ):
	"""
	@bug: 'quad' does not converge well with this function
	"""
	integrand = lambda x: np.sqrt( 1 + splev( x, s, 1 ) ** 2 )
	
	return quad( integrand, a, b , limit = 100 )[0]

# arclength_spline


def find_active_areas( x0: float, poly: np.poly1d, lengths, pix_per_mm ):
	''' Determines the active area x parameters for the fit polynomial given
		a desired arclength(s).
	'''
	
	lengths = pix_per_mm * np.array( lengths )

	ret_x = []
	for l in lengths:
		ret_x.append( find_param_along_poly( poly, x0, l )[0] )

	return ret_x

# find_active_areas


def fit_polynomial( centerline_img, deg ):
	nonzero = np.argwhere( centerline_img )
# 	x_coord = nonzero[:, 1]
# 	y_coord = nonzero[:, 0]
# 	poly = np.poly1d( np.polyfit( x_coord, y_coord, deg ) )
	
	N_rows, N_cols = np.shape( centerline_img )
	
	x = np.arange( N_cols )  # x-coords
	y = N_rows * np.ones( N_cols )  # y-coords
	y = np.argmax( centerline_img, 0 )

	x = x[y > 0]
	y = y[y > 0]
	poly = np.poly1d( np.polyfit( x, y, deg ) )

	return poly, x

# fit_polynomial


def fit_spline( centerline_img ):
	N_rows, N_cols = np.shape( centerline_img )
	
	x = np.arange( N_cols )  # x-coords
	y = N_rows * np.ones( N_cols )  # y-coords
	y = np.argmax( centerline_img, 0 )

	x = x[y > 0]
	y = y[y > 0]
	
# 	spline = splrep( x, y  )
	spline = CubicSpline( x, y, bc_type = "natural" )
	
	return spline, x
	
# fit_spline


def find_curvature( p: np.poly1d, x ):
	p1 = np.polyder( p, 1 )
	p2 = np.polyder( p, 2 )
	
	return p2( x ) / ( 1 + ( p1( x ) ) ** 2 ) ** ( 3 / 2 )

# find_curvature


def find_spline_curvature ( s, x ):
	if isinstance( s, tuple ):
		num = splev( x, s, 2 )
		denom = ( 1 + splev( x, s, 1 ) ** 2 ) ** ( 3 / 2 )
		retval = num / denom
		
	# if
	
	else:
		s1 = s.derivative( 1 )
		s2 = s.derivative( 2 )
		retval = s2( x ) / ( 1 + ( s1( x ) ) ** 2 ) ** ( 3 / 2 )
		
	# else
	
	return retval

# find_spline_curvature


def fit_circle_curvature( p, x, x_int, width: float ):
    k = []
    for xi in x_int:
        x_window = x[np.abs( x - xi) <= width]
        y_window = p( x_window )
        
        xm = np.mean( x_window )
        ym = np.mean( y_window )
        Ri = np.sqrt( ( x_window - xm ) ** 2 + ( y_window - ym ) ** 2 )  # distances
        
        calcdist = lambda xc, yc: np.sqrt( ( x_window - xc ) ** 2 + ( y_window - yc ) ** 2 )
        costfn = lambda c: np.abs( calcdist( *c ) - np.mean( calcdist( *c ) ) )
        
        c2, ier = leastsq( costfn , ( xm, ym ) )
        R = np.mean( calcdist( *c2 ) )
        
        k.append( 1/R )
        
    # for
    
    return np.array( k )

# fit_circle_curvature

def find_active_areas_poly( centerline_img, poly, pix_per_mm ):
	''' Starting with the tip and working backwards
	Using curvature calculation between pixels to determine incremental distance'''

	dist1 = 5 * pix_per_mm
	dist2 = 20 * pix_per_mm
	dist3 = 64 * pix_per_mm

	nonzero = np.argwhere( centerline_img )
	nonzero = sorted( nonzero, key = lambda element: element[1] )
	# import pdb; pdb.set_trace()

	x_tip = nonzero[-1][1]
	# print(x_tip)
	integrand = ( np.poly1d( [1] ) + np.poly1d.deriv( poly ) ** 2 ) ** 0.5
	print( type( integrand ) )
	integral = np.poly1d.integ( integrand )
	print( type( integral ) )
	tip_dist = np.polyval( integral, x_tip )
	print( tip_dist )

	current_idx = -1  # start at the tip
	prev_dist = 0
	while True:
		current_idx -= 1
		print( current_idx )
		lower_bound = nonzero[current_idx][1]

		current_dist = tip_dist - np.polyval( np.poly1d.integ( integrand ), lower_bound )
		print( current_dist )

		if prev_dist < dist1 and current_dist >= dist1:
			fbg1 = nonzero[current_idx]
			print( 'fbg1: %s' % fbg1 )

		if prev_dist < dist2 and current_dist >= dist2:
			fbg2 = nonzero[current_idx]
			print( 'fbg2: %s' % fbg2 )

		if prev_dist < dist3 and current_dist >= dist3:
			fbg3 = nonzero[current_idx]
			print( 'fbg3: %s' % fbg3 )
			break

		prev_dist = current_dist

	return fbg1, fbg2, fbg3

# find_active_areas_poly


def plot_func_image( img, func, x ):
	y = func( x )
	tempfile = "Output/temporary_img.png"
	result = cv2.imwrite( tempfile, img )
	
	if not result:
		raise OSError( "Image file was not written." )
	
	img = plt.imread( tempfile )
	os.remove( tempfile )
	
	plt.imshow( img , cmap = "gray" )
	plt.plot( x, y , 'r-' )
	plt.title( "Plot of function on image" )
	plt.show()
	
# plot_func_image


def plot_spline_image( img, s, x ):
	
	if isinstance( s, tuple ):
		y = splev( x, s )

	else:
		y = s( x )

	tempfile = "Output/temporary_img.png"
	result = cv2.imwrite( tempfile, img )
	
	if not result:
		raise OSError( "Image file was not written." )
	
	img = plt.imread( tempfile )
	os.remove( tempfile )
	
	plt.imshow( img , cmap = "gray" )
	plt.plot( x, y , 'r-' )
	plt.title( "Plot of function on image" )
	plt.show()
	
# plot_spline_image
	

def main():
	# filename = argv[0]
	filename = '60mm_70mm.png'
	directory = 'Test Images/Curvature_experiment_11-15-19/'
	pix_per_mm = 8.498439  # 767625596
	crop_area = ( 84, 250, 1280, 715 )

	img, gray_image = load_image( directory + filename )
	
	crop_img = set_ROI_box( gray_image, crop_area )
# 	cv2.imshow( "Cropped Image", crop_img )
	
	binary_img = cv2.threshold( crop_img, 100, 255, cv2.THRESH_BINARY_INV )[1]
# 	cv2.imshow( "Binarized Image", binary_img )
	
# 	canny_edges = canny_edge_detection( crop_img )
	skeleton = get_centerline( binary_img )
# 	cv2.imshow( "Skeletonized image", skeleton )
	
	stitch_img = stitch( skeleton, binary_img )
	
	print( 'fitting the polynomial' )
	poly, x = fit_polynomial( skeleton, 15 )
	s, _ = fit_spline( skeleton )
	
	total_length = arclength( poly, np.min( x ), np.max( x ) )
	
	lengths = ( np.arange( 1, 25 ) / 25 ) * total_length
	x_sol = find_active_areas( np.min( x ), poly, lengths, 1 )
	
	curvatures = find_spline_curvature( s, x_sol )
	
	for i, lk in enumerate( zip( lengths, curvatures ) ):
		l, k = lk
		print( "{:2d}: l = {:.3f}, k = {:.3f} 1/mm, r = {:.3f} mm".format( i + 1,
													l, k * pix_per_mm,
													abs( 1 / k / pix_per_mm ) ) )
	
	y_sol = poly( x_sol )	
	
	draw_img = cv2.cvtColor( crop_img, cv2.COLOR_GRAY2BGR )
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i, pt in enumerate( zip( x_sol, y_sol ) ):
		pt = tuple( np.round( pt ).astype( int ) )
		draw_img = cv2.circle( draw_img, pt, 5, [0, 0, 255], -1 )
		
		pt_text = ( pt[0] - 20, pt[1] - 10 )
		draw_img = cv2.putText( draw_img, str( i + 1 ), pt_text, font, 1, [0, 0, 255],
							2, cv2.LINE_AA )
		
	cv2.imshow( "Active Areas", draw_img )
	
	cv2.waitKey( 50 )
	
	plt.plot( x, 1 / find_spline_curvature( s, x ) / pix_per_mm )
	plt.ylim( -100, 100 )
	plt.title( "Radius of Curvature vs. x" )
	plt.figure()
# 	plt.show()
# 	plot_func_image( crop_img, poly, x )
	plot_spline_image( crop_img, s, x )
	
# 	cv2.waitKey( 0 )
	
	cv2.destroyAllWindows()
	
# main


def main_test_spline():
	x = np.arange( 201 ) / 100 - 1
	f = lambda x: np.sqrt( 1 - x ** 2 )
	
	s = splrep( x, f( x ) )
	
	k = find_spline_curvature( s, x )
	
	plt.figure( 1 )
	plt.plot( x, f( x ), 'k.', x, splev( x, s ), 'r-' )
	plt.title( "hemi circle plot" )
	
	plt.figure( 2 )
	plt.plot( x, 1 / k )
	plt.title( "Curvature Plot" )
	plt.ylabel( "Radius of Curvature" )
	plt.xlabel( "X" )
	
	plt.show()

# main_test_spline
	
	radii = 1/fit_circle_curvature( poly, x, x, pix_per_mm)/pix_per_mm
	plt.plot(x, radii)
	plt.show()

# main


def main_error():
	filename = 'S-shape_90mm_100mm.PNG'
	directory = 'Test Images/Solidworks_generated/'
	crop_area = ( 200, 200, 250, 250 )

	img, gray_image = load_image( directory + filename )
	crop_img = set_ROI( gray_image, crop_area )
	
	# binarize and invert
	thresh = 50
	crop_img[crop_img < thresh] = 0
	crop_img[crop_img != 0] = 255
	inverted_img = 255 - crop_img

	skeleton = get_centerline( inverted_img )
	nonzero = np.argwhere( skeleton )
	adj = np.amin( nonzero[:, 1] )
	skeleton_crop = set_ROI( skeleton, ( adj, 0, 0, 0 ) )
	print( np.amax( nonzero[:, 1] ) )
	cv2.imshow( 'skeleton', skeleton_crop )
	cv2.waitKey( 0 )

	poly_coeff = fit_polynomial( skeleton_crop, 7 ).c
	np.set_printoptions( precision = 10, suppress = True )
	print( poly_coeff )
	
# main_error


if __name__ == '__main__':
# 	main(sys.argv[1:])
	main()
# 	main_error()
	

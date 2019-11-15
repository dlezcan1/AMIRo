import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys
from scipy.integrate import quad
from scipy.optimize import fsolve
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
	x_coord = nonzero[:, 1]
	y_coord = nonzero[:, 0]
	poly = np.poly1d( np.polyfit( x_coord, y_coord, deg ) )

	return poly, x_coord


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


def main():
	# filename = argv[0]
	filename = '10mm_60mm_3mm.png'
	directory = 'Test Images/Curvature_experiment_10-28/'
	pix_per_mm = 8.498439  # 767625596
	crop_area = ( 50, 40, 420, 210 )

	img, gray_image = load_image( directory + filename )
	crop_img = set_ROI( gray_image, crop_area )
	binary_img = binary( crop_img )
	canny_edges = canny_edge_detection( crop_img )
	skeleton = get_centerline( canny_edges )
	stitch_img = stitch( skeleton, binary_img )
	# fbg1, fbg2, fbg3 = find_active_areas(stitch_img, pix_per_mm)
	# print('fbg1: %s' % fbg1)
	# print('fbg2: %s' % fbg2)
	# print('fbg3: %s' % fbg3)
	print( 'fitting the polynomial' )
	poly = fit_polynomial( stitch_img, 10 )
	fbg1_poly, fbg2_poly, fbg3_poly = find_active_areas_poly( stitch_img, poly, pix_per_mm )

	# ## overlay FBG locations on cropped color image
	# fbg_img = set_ROI(img)
	# cv2.circle(fbg_img, (fbg1[1], fbg1[0]), 3, (0,255,0), 2)
	# cv2.circle(fbg_img, (fbg2[1], fbg2[0]), 3, (0,255,0), 2)
	# cv2.circle(fbg_img, (fbg3[1], fbg3[0]), 3, (0,255,0), 2)

	# ## overlay skeleton centerline over cropped color image
	# fbg_img[stitch_img != 0] = (0,0,255)

	# cv2.imwrite('output/' + filename + '_gray.png', gray_image)
	# cv2.imwrite('output/' + filename + '_cropped.png', crop_img)
	# cv2.imwrite('output/' + filename + '_binary.png', binary_img)
	# cv2.imwrite('output/' + filename + '_canny.png', canny_edges)
	# cv2.imwrite('output/' + filename + '_skeleton.png', skeleton)
	# cv2.imwrite('output/' + filename + '_fbg.png', fbg_img)
	# cv2.imwrite('output/' + filename + '_stitch.png', stitch_img)


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


if __name__ == '__main__':
# 	main(sys.argv[1:])
# 	main()
# 	main_error()
	poly = np.poly1d( np.random.randn( 2 ) )
	
	xf = np.random.randn( 10 )
	x0 = np.min(xf);
	x = np.max( xf ) * np.arange( 100 ) / 100 + x0
	y0, *yf = poly( [x0, xf] )
	yf = yf[0]
	
	length = np.sqrt( ( yf - y0 ) ** 2 + ( xf - x0 ) ** 2 )
	
	solution = find_active_areas( x0, poly, length, 1 )
	
	print( "Initial x_f's:", np.round( xf, 2 ) )
	print( "Solution x_f's:", np.round( solution, 2 ) )
	print
	

import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys


def load_image( filename ):
	img = cv2.imread( filename + '.png', cv2.IMREAD_COLOR )
	gray_image = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

	return img, gray_image


def set_ROI( image ):
	''' Note that origin of images is typically top lefthand corner'''
	startX = 143
	endX = image.shape[1] - 40
	startY = 420
	endY = image.shape[0] - 210

	cropped = image[startY:endY, startX:endX]

	return cropped


def binary( image ):
	thresh_dark = 40
	thresh_light = 230
	binary_img = np.copy( image )
	binary_img[binary_img <= thresh_dark] = 0
	# binary_img[binary_img > thresh_light] = 0
	binary_img[binary_img != 0] = 255

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
	bor1 = [0, 26, 93, 165]  # xleft, xright, ytop, ybottom #for the bottomleft corner, blackout
	bor2 = [0, 29, 0, 58]  # xleft, xright, ytop, ybottom #for the topleft corner, blackout

	img = np.copy( image )

	# edges = cv2.Canny(image, thresh1, thresh2)

	# # Canny Filtering for Edge detection
	canny1 = cv2.Canny( img, thresh1, thresh2 )
	# cv2.imshow('canny1 before',canny1)

	# # Remove (pre-determined for simplicity in this code) artifacts manually
	# # I plan to make this part of the algorithm to be incorproated into GUI
	# canny1[bor1[2]:bor1[3],bor1[0]:bor1[1]] = 0
	# canny1[bor2[2]:bor2[3],bor2[0]:] = 0
	# #        cv2.imshow('canny1 after',canny1)

	# worked for black background
	kernel = gen_kernel( ( 7, 7 ) )
	canny1_fixed = cv2.morphologyEx( canny1, cv2.MORPH_CLOSE, kernel )
	kernel = gen_kernel( ( 9, 9 ) )
	canny1_fixed = cv2.dilate( canny1_fixed, kernel, iterations = 2 )
	kernel = gen_kernel( ( 11, 31 ) )
	canny1_fixed = cv2.erode( canny1_fixed, kernel, iterations = 1 )
	kernel = gen_kernel( ( 7, 7 ) )
	canny1_fixed = cv2.morphologyEx( canny1_fixed, cv2.MORPH_OPEN, kernel )
	canny1_fixed = cv2.erode( canny1_fixed, kernel, iterations = 1 )

	retval = canny1_fixed
	cv2.imshow( 'canny step', retval )

	return retval


def get_centerline( edge_image ):
	centerline_img = np.zeros( edge_image.shape )

	for col in range( edge_image.shape[1] ):
		nonzero = np.argwhere( edge_image[:, col] != 0 )

		if len( nonzero ) != 0:
			center_row = int( np.rint( np.mean( nonzero ) ) )
			centerline_img[center_row, col] = 255

	return centerline_img


def fit_polynomial( centerline_image, order ):
	assert( np.max( centerline_image ) <= 1 )
	
	N_rows, N_cols = np.shape( centerline_image )
	
	x = np.arange( N_cols )  # x-coords
	y = N_rows * np.ones( N_cols )  # y-coords
	y = np.argmax( centerline_image, 0 )
	
	x = x[y > 0]
	y = y[y > 0]
	
	if len( x ) == 0:
		x = np.zeros( N_cols )
		y = np.zeros( N_cols )
		p = np.poly1d( [0] )
	else:
		p = np.poly1d( np.polyfit( x, y, order ) )
	
	return x, y, p

# fit_polynomial


def find_active_areas( centerline_image, pix_per_mm ): 
	area1 = 5 * pix_per_mm
	area2 = 20 * pix_per_mm
	area3 = 64 * pix_per_mm
	x1, x2, x3 = 0, 0, 0  # x-positions of the areas of interest
	d1, d2, d3 = np.inf * np.ones( 3 )
	ds = 0.05 * pix_per_mm

	centerline_image[centerline_image > 0] = 1  # convert to ones
	x, _, f = fit_polynomial( centerline_image )

# 	f_p = np.polyder( f )  # df/dx

	l_prev, l_curr = 0, 0
	x_prev, y_prev = x[-1], f( x[-1] )
	x_curr, y_curr = x_prev, y_prev
	while l_prev <= area3:
		x_curr += ds
		y_curr = f( x_curr )
		l_curr = l_prev + np.linalg.norm( [x_curr - x_prev, y_curr - y_prev] )
		
		if ( l_prev <= area1 and area1 <= l_curr and np.abs( l_curr - area1 ) < d1 ):
			x1 = x_curr
		
		if ( l_prev <= area2 and area2 <= l_curr and np.abs( l_curr - area2 ) < d2 ):
			x2 = x_curr
		
		if ( l_prev <= area3 and area3 <= l_curr and np.abs( l_curr - area3 ) < d3 ):
			x3 = x_curr
			break
	# while
	
	return x1, x2, x3
# find_active_areas


def find_curvature( centerline_img ):
	circles = cv2.HoughCircles( centerline_img, cv2.HOUGH_GRADIENT, 1, 20,
		param1 = 225, param2 = 30, minRadius = 0, maxRadius = 1000 )

	# # draw the circles
	img_circles = np.copy( centerline_img )
	for circle in circles:
		cv2.circle( img_circles, ( circle[0], circle[1] ), circle[2],
			( 0, 255, 0 ), thickness = 1 )

	return img_circles


def compare_with_original( original_image, filtered_image, origin ):
	# location is an array with the coordinates of the filtered image origin [x_start, y_start]
	padded_img = np.zeros( original_image.shape )

	img_size = filtered_image.shape
	padded_img[origin[1]:origin[1] + img_size[1] + 1,
		origin[0]:origin[0] + img_size[0] + 1] = np.copy( filtered_image )

	alpha = 0.5
	image_stitched = cv2.addWeighted( original_image, alpha, padded_img, 1 - alpha, 0.0 )

	return image_stitched


def dilate_img( edge_image, kernelSize ):
	# kernelSize is a tuple

	kernel = np.ones( kernelSize, np.uint8 )
	dilated_img = cv2.dilate( edge_image, kernel, iterations = 5 )

	return dilated_img


def erode_img( img_dilated, kernelSize ):
	# kernelSize is a tuple

	kernel = np.ones( ( 3, 3 ), np.uint8 )
	eroded_img = cv2.erode( img_dilated, kernel, iterations = 12 )

	return eroded_img


def main():
	# filename = argv[0]
	filename = 'image5-test-color-s_shape.png'
	directory = 'Test Images/'

	img, gray_image = load_image( directory + filename )
	crop_img = set_ROI( gray_image )
	# binary_img = binary(crop_img)
	canny_edges = canny_edge_detection( crop_img )
	# img_dilated = dilate_img(canny_edges)
	# img_eroded = erode_img(img_dilated)
	centerline_img = get_centerline( canny_edges )
	# img_circles = find_circles(centerline_img)
	stitched_img = compare_with_original( img, canny_edges, ( 143, 420 ) )

	# cv2.imwrite('output/' + filename + '_gray.png', gray_image)
	# cv2.imwrite('output/' + filename + '_cropped.png', crop_img)
	# cv2.imwrite('output/' + filename + '_binary.png', binary_img)
	cv2.imwrite( 'output/' + filename + '_canny.png', canny_edges )
	# cv2.imwrite('output/' + filename + '_dilated.png', img_dilated)
	# cv2.imwrite('output/' + filename + '_eroded.png', img_eroded)
	cv2.imwrite( 'output/' + filename + '_centerline.png', centerline_img )
	# cv2.imwrite('output/' + filename + '_centerline.png', img_circles)
	cv2.imwrite( 'output/' + filename + '_compare.png', stitched_img )


if __name__ == '__main__':
	main( sys.argv[1:] )

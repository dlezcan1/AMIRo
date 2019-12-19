import numpy as np
import cv2
import image_processing as img_proc
import matplotlib.pyplot as plt
import p1_object_attributes as p1
import itertools


def set_ROI( image ):
	''' Note that origin of images is typically top lefthand corner'''
	# # circle grid
	# startX = 60
	# endX = image.shape[1] - 65
	# startY = 360
	# endY = image.shape[0] - 300

	# # square grid
	startX = 120
	endX = image.shape[1] - 135
	startY = 405
	endY = image.shape[0] - 370

	cropped = image[startY:endY, startX:endX]
	# cv2.imshow('cropped', cropped)
	# cv2.waitKey(0)
	return cropped


def invert_binary( cropped_img ):
	thresh = 100
	binary_img = np.copy( cropped_img )
	cv2.imshow( 'original', binary_img )
	binary_img = cv2.bilateralFilter( binary_img, 3, sigmaColor = 100, sigmaSpace = 100 )
	cv2.imshow( 'bilateralFilter', binary_img )

	binary_img[binary_img < thresh] = 0
	binary_img[binary_img != 0] = 255

	cv2.imshow( 'binary_img', binary_img )
	cv2.waitKey( 0 )

	invert_img = 255 - binary_img

	return invert_img


def baseline_distances( num_cols, num_rows, space ):
	''' Inputs: num_cols = number of columns of circles
				num_rows = number of rows of circles
				space = spacing between circles (center to center) in mm
		Output: num_rows x num_cols numpy array of distances in mm
				origin is the top left corner'''

	dist_baseline = np.zeros( ( num_rows, num_cols ) )
	for r in range( num_rows ):
		for c in range( num_cols ):
			dist_baseline[r, c] = space * np.sqrt( r ** 2 + c ** 2 )

	# print(dist_baseline)

	return dist_baseline


def measure_distances( center_points ):
	""" returns the distance from the center of all of the points """
	# print(center_points)
	centroid = np.mean( center_points, axis = 0 )
	centroid_idx = np.argmin( np.linalg.norm( center_points - centroid, axis = 1 ) ) 
	centroid = center_points[ centroid_idx, : ]
	
	distances = np.linalg.norm( center_points - centroid, axis = 1 )
	distances = distances.reshape( 5, 30 )
	
	y_center_idx, x_center_idx = np.unravel_index( centroid_idx,
	                                               distances.shape )
	print( x_center_idx )
	print( y_center_idx )
	
	return distances, x_center_idx, y_center_idx

# measure_distances


def expected_distances( center_indices: tuple ):
	pix_to_mm = 8.498439767625596
	pix_spacing = 4 * pix_to_mm 
	x_center_idx, y_center_idx = center_indices
	v_distances = np.arange( -x_center_idx, 30 - x_center_idx )
	h_distances = np.arange( -y_center_idx, 5 - y_center_idx )
	
	exp_distances = np.array( [np.linalg.norm( d ) for d in
	                            itertools.product( h_distances, v_distances )] )
	exp_distances *= pix_spacing
	
	return ( exp_distances )

# expected_distances


def get_centers( cropped_img, num_cols, num_rows ):
	measured_centers = np.zeros( ( num_rows, num_cols ) )
	circles = cv2.HoughCircles( cropped_img, cv2.HOUGH_GRADIENT, dp = 1,
		minDist = 3, param1 = 50, param2 = 15, minRadius = 10, maxRadius = 15 )

	radius = np.mean( circles[0][:, 2] )
	# print(circles[0])
	# print(circles[0].shape)
	sorted_circles = sorted( circles[0][:, 0:2], key = lambda element: ( element[1], element[0] ) )
	sorted_circles = np.asarray( sorted_circles ).reshape( ( num_rows, num_cols, 2 ) )
	# print(sorted_circles)
	# print(sorted_circles.shape)

	# for r in range(num_rows):
	# 	fig = plt.figure()
	# 	plt.plot(sorted_circles[r,:,0], sorted_circles[r,:,1])
	# 	plt.show()
	
	return radius, sorted_circles


def get_centers_segment( invert_img, num_cols, num_rows ):
	labeled_image = p1.label( invert_img )
	cv2.imwrite( 'output/image_processing_binary_image_squares_labeled.png', labeled_image )
	center_list = np.empty( ( 0, 2 ) )

	# # compute centroids
	labels, counts = np.unique( labeled_image, return_counts = True )
	labels = labels[1:]  # remove the 0 for background
	counts = counts[1:]

	for obj in range( len( labels ) ):
		area = counts[obj]

		# calculate position
		idx = np.argwhere( labeled_image == labels[obj] )
		x_sum = np.sum( idx[:, 1] )
		y_sum = np.sum( idx[:, 0] )

		x_pos = x_sum / area
		y_pos = y_sum / area

		# print([y_pos,x_pos])

		center_list = np.vstack( [center_list, [y_pos, x_pos]] )

	# attribute_list = p1.get_attribute(labeled_image)

	# center_list = np.empty((0,2))
	# for d in attribute_list:
	# 	center = d["position"]
	# 	x = center["x"]
	# 	y = invert_img.shape[0] - center["y"]
	# 	center_list = np.vstack([center_list, [y,x]])

	sorted_centers = sorted( center_list, key = lambda element: element[0] )
	sorted_centers = np.asarray( sorted_centers ).reshape( ( num_rows, num_cols, 2 ) )
	# sorted_centers = np.asarray(center_list).reshape((num_rows,num_cols,2))
	for r in range( num_rows ):
		row_list = sorted_centers[r, :, :].tolist()
		row_list = sorted( row_list, key = lambda element: element[1] )
		row_array = np.asarray( row_list )
		sorted_centers[r, :, :] = row_array
	# print(sorted_centers)
	# print(invert_img.shape)
	
	return sorted_centers


def plot_error( dist_baseline, sorted_circles ):
	dist_measured = np.zeros( dist_baseline.shape )
	origin = sorted_circles[0, 0, :]

	for r in range( sorted_circles.shape[0] ):
		for c in range( sorted_circles.shape[1] ):
			dist_measured[r, c] = np.linalg.norm( sorted_circles[r, c, :] - origin )

	# error = dist_measured - dist_baseline
	pix_to_mm = dist_measured[dist_measured != 0] / dist_baseline[dist_baseline != 0]
		
	pix_to_mm = np.mean( pix_to_mm )
	error = dist_measured / pix_to_mm - dist_baseline

	# # filter out the outliers
	# error_filtered = error[error < 3]
	# error_filtered = error_filtered[error_filtered > -3]
	# print('mean error: %s' % np.mean(error))
	# print('max error: %s' % np.amax(error))
	# print('min error: %s' % np.amin(error))

	# # plot error vs. distance
	rows = dist_baseline.shape[0]
	cols = dist_baseline.shape[1]

	# # plot with all points together
	# error_flat = error.reshape((rows*cols,))
	# dist_flat = dist_baseline.reshape((rows*cols,))

	# fig = plt.figure()
	# plt.title('Error vs. Distance (mm)')
	# plt.plot(dist_flat, error_flat, 'b.')
	# plt.xlabel('Distance (mm)')
	# plt.ylabel('Error (measured - baseline) (mm)')

	# plot by row
	color_list = ['b', 'c', 'g', 'm', 'r']
	fig = plt.figure()
	plt.title( 'Error vs. Distance (mm)' )
	plt.xlabel( 'Distance (mm)' )
	plt.ylabel( 'Error (measured - baseline) (mm)' )
	text = 'mean error: %.3f \n max error: %.3f \n min error: %.3f' % ( np.mean( error ), np.amax( error ), np.amin( error ) )
	# print(text)
	plt.text( 0, -0.55, text )

	for r in range( rows ):
		marker = color_list[r] + '.'
		label = 'row ' + str( r )
		plt.plot( dist_baseline[r, :], error[r, :], marker, label = label )

	plt.legend()

	# plot a zero line
	x = [0, dist_baseline[-1, -1]]
	y = [0, 0]
	plt.plot( x, y, color = 'r' )

	plt.show()

	return error


def plot_error_center( num_rows, num_cols, measured_distances, expected_distances ):
	# error = dist_measured - dist_baseline
	pix_to_mm = 8.498439767625596
	dist_baseline = expected_distances.reshape( ( num_rows, num_cols ) )
	error = ( measured_distances - dist_baseline ) / pix_to_mm

	# # plot error vs. distance
	rows = dist_baseline.shape[0]
	cols = dist_baseline.shape[1]

	# # plot with all points together
	# error_flat = error.reshape((rows*cols,))
	# dist_flat = dist_baseline.reshape((rows*cols,))

	# fig = plt.figure()
	# plt.title('Error vs. Distance (mm)')
	# plt.plot(dist_flat, error_flat, 'b.')
	# plt.xlabel('Distance (mm)')
	# plt.ylabel('Error (measured - baseline) (mm)')

	# plot by row
	color_list = ['b', 'c', 'g', 'm', 'r']
	fig = plt.figure()
	plt.title( 'Error vs. Distance (mm)' )
	plt.xlabel( 'Distance (mm)' )
	plt.ylabel( 'Error (measured - baseline) (mm)' )
	text = 'mean error: %.3f \n max error: %.3f \n min error: %.3f' % ( np.mean( error ), np.amax( error ), np.amin( error ) )
	# print(text)
	plt.text( 0, -0.55, text )

	for r in range( rows ):
		marker = color_list[r] + '.'
		label = 'row ' + str( r )
		plt.plot( dist_baseline[r, :], error[r, :], marker, label = label )

	plt.legend()

	# plot a zero line
	x = [0, dist_baseline[-1, -1]]
	y = [0, 0]
	plt.plot( x, y, color = 'r' )

	plt.show()

	return error


def main():
	# filename = argv[0]
	# filename = '5x25_circles_D-2.5mm_space-5mm.png'
	filename = 'image_processing_binary_image_squares'
	directory = 'Test Images/'

	num_cols = 30
	num_rows = 5
	center_separation = 4

	img, gray_image = img_proc.load_image( directory + filename + '.png' )
	# crop_img = set_ROI(gray_image)
	# invert_img = invert_binary(crop_img)
	dist_baseline = baseline_distances( num_cols, num_rows, center_separation )
	# radius, circles = get_centers(crop_img, 25, 5)

	sorted_centers = get_centers_segment( gray_image, num_cols, num_rows )

	measured_distances, x_center_idx, y_center_idx = measure_distances( sorted_centers.reshape( ( -1, 2 ) ) )
	expected_dist = expected_distances( ( x_center_idx, y_center_idx ) )

	# ## marked circle image
	# circle_img = set_ROI(img)
	# for c in circles:
	# 	for r in range(25):
	# 		center = (c[r,0], c[r,1])
	# 		cv2.circle(circle_img, center, radius, (0,255,0), 1)
	# 		cv2.circle(circle_img, center, 1, (0,255,0), 2)
	# origin = circles[0,0,:]
	# cv2.circle(circle_img, (origin[0], origin[1]), 2, (125,125,0), 2)
	# # cv2.imshow('test', test_img)
	# # cv2.waitKey(0)

	# # marked square image
	square_img = np.copy( img )
	colors = [( 255, 0, 0 ), ( 0, 255, 0 ), ( 0, 0, 255 ), ( 125, 125, 0 ), ( 0, 125, 125 )]
	count = 0
	for c in sorted_centers:
		for r in range( num_cols ):
			square_img[int( np.round( c[r, 0] ) ), int( np.round( c[r, 1] ) )] = colors[count]
			# center = (int(c[r,1]), int(c[r,0]))
			# cv2.circle(square_img, center, 2, colors[count], 2)
		count += 1
	# cv2.imshow('square_img', square_img)
	# cv2.waitKey(0)
	cv2.imwrite( 'output/' + filename + '_square.png', square_img )

	# error = plot_error(dist_baseline, sorted_centers)
	error = plot_error_center( num_rows, num_cols, measured_distances, expected_dist )
	# print(error)
	# outliers = np.argwhere(np.abs(error) > 3)
	# # print(outliers.shape)
	# for out in outliers:
	# 	center = circles[out[0], out[1], :]
	# 	cv2.circle(circle_img, (center[0], center[1]), 1, (0,0,255), 2)

	# cv2.imwrite('output/' + filename + '_circles.png', circle_img)
	# cv2.imwrite('output/' + filename + '_cropped.png', crop_img)
	# cv2.imwrite('output/' + filename + '_invertbinary.png', invert_img)


if __name__ == '__main__':
	main()

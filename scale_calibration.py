import cv2
import numpy as np
import sys

def load_image(filename):
	img = cv2.imread(filename, cv2.IMREAD_COLOR)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return img, gray_image

def set_ROI(image):
	''' Note that origin of images is typically top lefthand corner'''
	startX = 143
	endX = image.shape[1] - 140
	startY = 200
	endY = image.shape[0] - 100

	cropped = image[startY:endY, startX:endX]

	return cropped

def process_image(cropped_img):
	# ## Invert the colors
	# inverted_img = 255-cropped_img

	# ## Closing to get rid of artifacts
	# kernel = np.ones((5,5))
	thresh = 200
	# temp = cv2.morphologyEx(inverted_img, cv2.MORPH_CLOSE, kernel)

	# orig_color = 255-temp
	orig_color = np.copy(cropped_img)

	orig_color[orig_color > thresh] = 255
	orig_color[orig_color <= thresh] = 0

	result = orig_color
	return result

def get_scale(image):
	## assumes that we are using a 1cm checkerboard

	num_cols = 7 # x-direction
	num_rows = 5 # y-direction
	retval, corners = cv2.findChessboardCorners(image, (num_rows,num_cols), None)
	corners = corners.reshape((num_cols,num_rows,2))
	# print(corners)

	avg_per_row = np.zeros(num_rows)
	avg_per_col = np.zeros(num_cols)

	for r in range(num_rows):
		dist_sum = 0
		for c in range(1, num_cols):
			dist_sum += np.linalg.norm(corners[c,r,:] - corners[c-1,r,:])

		avg_per_row[r] = dist_sum/(num_cols-1)

	for c in range(num_cols):
		dist_sum = 0
		for r in range(1, num_rows):
			dist_sum += np.linalg.norm(corners[c,r-1,:] - corners[c,r,:])

		avg_per_col[c] = dist_sum/(num_rows-1)

	# print('avg_per_row: %s' % avg_per_row)
	# print('avg_per_col: %s' % avg_per_col)

	avg_pixels_per_mm = np.mean(np.hstack((avg_per_row, avg_per_col)))/10.
	print('avg_pixels_per_mm: %s' % avg_pixels_per_mm)

	return avg_pixels_per_mm

def check_scale(image, pixels_per_mm):
	## Find the circles in the image
	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 2, 100)
	ref_radius = 5.

	## compute the error
	error = 0
	for c in circles[0]:
		error += c[2] - pixels_per_mm*ref_radius

	error = error/len(circles)
	print('average error in circle radius measurement in pixels: %s' % error)

	return circles[0], error

def main():
	# filename = argv[0]
	filename = 'scaling_calibration_10-23-2019.png'
	directory = 'Test Images/'

	img, gray_image = load_image(directory + filename)
	crop_img = set_ROI(gray_image)
	# proc_img = process_image(crop_img)
	pixels_per_mm = get_scale(crop_img)
	circles, error = check_scale(crop_img, pixels_per_mm)
	
	## if outputing corners, draw the corners on color image
	# corners = corners.reshape((35,2))
	# corner_img = set_ROI(img)
	# for c in corners:
	# 	cv2.circle(corner_img, (c[0],c[1]), 4, (0,255,0), 1)

	## draw circles on image
	check_img = set_ROI(img)
	for c in circles:
		radius = np.round(c[2]).astype('int')
		cv2.circle(check_img, (c[0],c[1]), radius, (0,255,0), 2)
	
	# cv2.imwrite('output/' + filename + '_gray.png', gray_image)
	cv2.imwrite('output/' + filename + '_cropped.png', crop_img)
	# cv2.imwrite('output/' + filename + '_processed.png', proc_img)
	cv2.imwrite('output/' + filename + '_circle-check.png', check_img)
	# cv2.imwrite('output/' + filename + '_dilated.png', img_dilated)
	# cv2.imwrite('output/' + filename + '_eroded.png', img_eroded)
	# cv2.imwrite('output/' + filename + '_centerline.png', centerline_img)
	# cv2.imwrite('output/' + filename + '_centerline.png', img_circles)
	# cv2.imwrite('output/' + filename + '_compare.png', stitched_img)

if __name__ == '__main__':
	main()
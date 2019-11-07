import numpy as np
import cv2
import image_processing as img_proc
import matplotlib.pyplot as plt

def set_ROI(image):
	''' Note that origin of images is typically top lefthand corner'''
	startX = 60
	endX = image.shape[1] - 65
	startY = 360
	endY = image.shape[0] - 300

	cropped = image[startY:endY, startX:endX]
	return cropped

def baseline_distances(num_cols, num_rows, space):
	''' Inputs: num_cols = number of columns of circles
				num_rows = number of rows of circles
				space = spacing between circles (center to center) in mm
		Output: num_rows x num_cols numpy array of distances in mm
				origin is the top left corner'''

	dist_baseline = np.zeros((num_rows, num_cols))
	for r in range(num_rows):
		for c in range(num_cols):
			dist_baseline[r,c] = space*np.sqrt(r**2 + c**2)

	# print(dist_baseline)

	return dist_baseline

def get_centers(cropped_img, num_cols, num_rows):
	measured_centers = np.zeros((num_rows, num_cols))
	circles = cv2.HoughCircles(cropped_img, cv2.HOUGH_GRADIENT, dp=1, 
		minDist=3, param1=50, param2=15, minRadius=10, maxRadius=15)

	radius = np.mean(circles[0][:,2])
	# print(circles[0])
	# print(circles[0].shape)
	sorted_circles = sorted(circles[0][:,0:2], key=lambda element: (element[1], element[0]))
	sorted_circles = np.asarray(sorted_circles).reshape((num_rows,num_cols,2))
	# print(sorted_circles)
	# print(sorted_circles.shape)

	# for r in range(num_rows):
	# 	fig = plt.figure()
	# 	plt.plot(sorted_circles[r,:,0], sorted_circles[r,:,1])
	# 	plt.show()
	
	return radius, sorted_circles

def plot_error(dist_baseline, sorted_circles):
	dist_measured = np.zeros(dist_baseline.shape)
	origin = sorted_circles[0,0,:]

	for r in range(sorted_circles.shape[0]):
		for c in range(sorted_circles.shape[1]):
			dist_measured[r,c] = np.linalg.norm(sorted_circles[r,c,:] - origin)

	# error = dist_measured - dist_baseline
	pix_to_mm = dist_measured[dist_measured != 0]/dist_baseline[dist_baseline != 0]
		
	pix_to_mm = np.mean(pix_to_mm)
	error = dist_measured/pix_to_mm - dist_baseline

	## filter out the outliers
	# error_filtered = error[error < 3]
	# error_filtered = error_filtered[error_filtered > -3]
	# print('mean error: %s' % np.mean(error))
	# print('max error: %s' % np.amax(error))
	# print('min error: %s' % np.amin(error))

	## plot error vs. distance
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
	plt.title('Error vs. Distance (mm)')
	plt.xlabel('Distance (mm)')
	plt.ylabel('Error (measured - baseline) (mm)')
	text = 'mean error: %s \n max error: %s \n min error: %s' % (np.mean(error), np.amax(error), np.amin(error))
	# print(text)
	plt.text(0,-0.55,text)

	for r in range(rows):
		marker = color_list[r] + '.'
		label = 'row ' + str(r)
		plt.plot(dist_baseline[r,:], error[r,:], marker, label=label)

	plt.legend()

	# plot a zero line
	x = [0, dist_baseline[-1,-1]]
	y = [0, 0]
	plt.plot(x, y, color='r')

	plt.show()

	return error


def main():
	# filename = argv[0]
	filename = '5x25_circles_D-2.5mm_space-5mm.png'
	directory = 'Test Images/'

	img, gray_image = img_proc.load_image(directory + filename)
	crop_img = set_ROI(gray_image)
	dist_baseline = baseline_distances(25, 5, 5)
	radius, circles = get_centers(crop_img, 25, 5)

	## marked circle image
	circle_img = set_ROI(img)
	for c in circles:
		for r in range(25):
			center = (c[r,0], c[r,1])
			cv2.circle(circle_img, center, radius, (0,255,0), 1)
			cv2.circle(circle_img, center, 1, (0,255,0), 2)
	origin = circles[0,0,:]
	cv2.circle(circle_img, (origin[0], origin[1]), 2, (125,125,0), 2)
	# cv2.imshow('test', test_img)
	# cv2.waitKey(0)

	error = plot_error(dist_baseline, circles)
	# print(error)
	outliers = np.argwhere(np.abs(error) > 3)
	# print(outliers.shape)
	for out in outliers:
		center = circles[out[0], out[1], :]
		cv2.circle(circle_img, (center[0], center[1]), 1, (0,0,255), 2)

	cv2.imwrite('output/' + filename + '_circles.png', circle_img)



if __name__ == '__main__':
	main()
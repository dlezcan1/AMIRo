import cv2
import numpy as np
from skimage.morphology import skeletonize
import sys

def load_image(filename):
	img = cv2.imread(filename, cv2.IMREAD_COLOR)
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return img, gray_image


def set_ROI(image):
	''' Note that origin of images is typically top lefthand corner'''
	startX = 50
	endX = image.shape[1] - 40
	startY = 420
	endY = image.shape[0] - 210

	cropped = image[startY:endY, startX:endX]

	return cropped

def binary(image):
	thresh_dark = 210
	thresh_light = 230
	binary_img = np.copy(image)
	binary_img[binary_img <= thresh_dark] = 0
	# binary_img[binary_img > thresh_light] = 0
	binary_img[binary_img != 0] = 255

	bor1 = [300,800,0,65] #xleft, xright, ytop, ybottom for the top, blackout
	binary_img[bor1[2]:bor1[3],bor1[0]:bor1[1]] = 0

	skeleton = np.copy(binary_img)/255
	skeleton = skeletonize(skeleton)
	binary_img = skeleton.astype(np.uint8)*255

	# cv2.imshow('binary', binary_img)
	# cv2.waitKey(0)

	return binary_img

def gen_kernel(shape):
	"""Function to generate the shape of the kernel for image processing

	@param shape: 2-tuple (a,b) of integers for the shape of the kernel
	@return: returns axb numpy array of value of 1's of type uint8
	"""
	return np.ones(shape,np.uint8)

def canny_edge_detection(image):
	thresh1 = 25
	thresh2 = 225
	bor1 = [300,800,0,65] #xleft, xright, ytop, ybottom for the top, blackout
	bor2 = [700,930,90,image.shape[0]] #xleft, xright, ytop, ybottom for the bottom, blackout

	img = np.copy(image)

	# edges = cv2.Canny(image, thresh1, thresh2)

	## Canny Filtering for Edge detection
	canny1 = cv2.Canny(img, thresh1, thresh2)
	# cv2.imshow('canny1 before',canny1)

	## Remove (pre-determined for simplicity in this code) artifacts manually
	## I plan to make this part of the algorithm to be incorproated into GUI
	canny1[bor1[2]:bor1[3],bor1[0]:bor1[1]] = 0
	canny1[bor2[2]:bor2[3],bor2[0]:bor2[1]] = 0
	cv2.imshow('canny1 after',canny1)
	# cv2.waitKey(0)

	# worked for black background
	kernel = gen_kernel((7,7))
	canny1_fixed = cv2.morphologyEx(canny1,cv2.MORPH_CLOSE,kernel)
	cv2.imshow('canny1 morph_close',canny1_fixed)

	kernel = gen_kernel((9,9))
	canny1_fixed = cv2.dilate(canny1_fixed,kernel,iterations=2)
	cv2.imshow('canny1 dilate',canny1_fixed)

	kernel = gen_kernel((11,31))
	canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)
	cv2.imshow('canny1 erode',canny1_fixed)

	kernel = gen_kernel((7,7))
	canny1_fixed = cv2.morphologyEx(canny1_fixed,cv2.MORPH_OPEN,kernel)
	cv2.imshow('canny1 morph_open',canny1_fixed)

	canny1_fixed = cv2.erode(canny1_fixed,kernel,iterations=1)
	cv2.imshow('canny1 erode2',canny1_fixed)
	cv2.waitKey(0)

	retval = canny1_fixed

	return retval

def get_centerline(edge_image):
	# centerline_img = np.zeros(edge_image.shape)

	# for col in range(edge_image.shape[1]):
	# 	nonzero = np.argwhere(edge_image[:,col] != 0)

	# 	if len(nonzero) != 0:
	# 		center_row = int(np.rint(np.mean(nonzero)))
	# 		centerline_img[center_row, col] = 255

	# try skeletonize
	binary = np.copy(edge_image)/255
	skeleton = skeletonize(binary)
	skeleton = skeleton.astype(np.uint8)*255

	return skeleton

def stitch(canny_img, binary_img):
	binary_nonzero = np.argwhere(binary_img) # find all nonzero indices

	# sort by x-coordinate
	binary_nonzero = sorted(binary_nonzero, key=lambda element: element[1])
	binary_nonzero = np.asarray(binary_nonzero) # convert back to numpy array

	canny_nonzero = np.argwhere(canny_img) # find all nonzero indices
	canny_max_idx = np.argmax(canny_nonzero[:,1]) # find the index of rightmost point
	canny_max = canny_nonzero[canny_max_idx][1] # find the index in original image

	stitch_img = np.copy(canny_img)
	# find where rightmost point stops in binary image
	binary_start = np.argwhere(binary_nonzero[:,1] == canny_max)[0][0]
	
	# add binary_img points to stitch_img
	for b in range(binary_start, binary_nonzero.shape[0]):
		coord = binary_nonzero[b]
		stitch_img[coord[0],coord[1]] = 255

	return stitch_img


def find_active_areas(centerline_image):
	''' Starting with the tip and working backwards
	Using L2 norm between pixels to determine incremental distance'''
	img = np.copy(centerline_image)

	pix_per_mm = 6.79#4541056950886
	dist1 = 5*pix_per_mm
	dist2 = 20*pix_per_mm
	dist3 = 64*pix_per_mm

	nonzero = np.argwhere(centerline_image)
	# nonzero[nonzero[:,1].argsort()] # sort by the second column, corresponding to x-coord
	nonzero = sorted(nonzero, key=lambda element: element[1])
	
	# import pdb; pdb.set_trace() 

	# pixel locations of FBGs, rounded to the lower column value
	prev_idx = -1 # start at the tip
	total_dist = 0
	while total_dist < dist3:
		current_idx = prev_idx - 1
		prev_pix = nonzero[prev_idx]
		current_pix = nonzero[current_idx]

		prev_idx = current_idx
		dist_step = np.linalg.norm(current_pix - prev_pix)
		# dist_step = 1

		if total_dist < dist1 and total_dist + dist_step >= dist1:
			fbg1 = current_pix

		if total_dist < dist2 and total_dist + dist_step >= dist2:
			fbg2 = current_pix

		total_dist += dist_step

	fbg3 = nonzero[prev_idx - 1]

	return fbg1, fbg2, fbg3

def fit_polynomial(centerline_img, deg):
	nonzero = np.argwhere(centerline_img)
	x_coord = nonzero[:,1]
	y_coord = nonzero[:,0]
	poly = np.poly1d(np.polyfit(x_cord, y_coord, deg))

	return poly

def find_active_areas_poly(centerline_img, poly):
	''' Starting with the tip and working backwards
	Using L2 norm between pixels to determine incremental distance'''

	pix_per_mm = 6.794541#056950886
	dist1 = 5*pix_per_mm
	dist2 = 20*pix_per_mm
	dist3 = 64*pix_per_mm

	x_tip = np.argmax(np.argwhere(centerline_img)[:,1])

	y_pix = np.polyval()


def find_curvature(centerline_img, fbg1, fbg2, fbg3):
	''' Use least squares fit to find the radius and curvature at each active area'''
	window = 5




def main():
	# filename = argv[0]
	filename = '10mm_60mm_3mm.png'
	directory = 'Test Images/Curvature_experiment_10-28/'

	img, gray_image = load_image(directory + filename)
	crop_img = set_ROI(gray_image)
	binary_img = binary(crop_img)
	canny_edges = canny_edge_detection(crop_img)
	skeleton = get_centerline(canny_edges)
	stitch_img = stitch(skeleton, binary_img)
	fbg1, fbg2, fbg3 = find_active_areas(stitch_img)
	print('fbg1: %s' % fbg1)
	print('fbg2: %s' % fbg2)
	print('fbg3: %s' % fbg3)

	## overlay FBG locations on cropped color image
	fbg_img = set_ROI(img)
	cv2.circle(fbg_img, (fbg1[1], fbg1[0]), 3, (0,255,0), 2)
	cv2.circle(fbg_img, (fbg2[1], fbg2[0]), 3, (0,255,0), 2)
	cv2.circle(fbg_img, (fbg3[1], fbg3[0]), 3, (0,255,0), 2)

	## overlay skeleton centerline over cropped color image
	fbg_img[stitch_img != 0] = (0,0,255)

	

	# cv2.imwrite('output/' + filename + '_gray.png', gray_image)
	# cv2.imwrite('output/' + filename + '_cropped.png', crop_img)
	# cv2.imwrite('output/' + filename + '_binary.png', binary_img)
	# cv2.imwrite('output/' + filename + '_canny.png', canny_edges)
	# cv2.imwrite('output/' + filename + '_skeleton.png', skeleton)
	# cv2.imwrite('output/' + filename + '_fbg.png', fbg_img)
	# cv2.imwrite('output/' + filename + '_stitch.png', stitch_img)

if __name__ == '__main__':
	# main(sys.argv[1:])
	main()
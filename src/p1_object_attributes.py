#!/usr/bin/env python3
import cv2
import numpy as np
import sys


def binarize(gray_image, thresh_val):
  # TODO: 255 if intensity >= thresh_val else 0

  binary_image = np.copy(gray_image)
  binary_image[binary_image >= thresh_val] = 255
  binary_image[binary_image != 255] = 0

  return binary_image

def label(binary_image):
  # TODO
  label = 1
  parent = [0]

  # create a padded image, which is the same as the binary image but with 
  # a border of zeros
  pad_img = np.zeros((binary_image.shape[0]+2, binary_image.shape[1]+2), dtype=int)
  nonzero = np.argwhere(binary_image != 0)

  for pt in nonzero:
    row = pt[0]
    col = pt[1]
    pad_img[row+1, col+1] = 255

  # cv2.imwrite('output/padded_image.png', pad_img)
  # labeled_image = np.copy(binary_image)

  ## Pass 1
  for r in range(0, pad_img.shape[0]):
    for c in range(0, pad_img.shape[1]):
      if pad_img[r,c] != 0: #if pixel is black background, leave it
        idx = (r,c)
        nbr = get_prior_neighbors(idx)

        n = nbr[0]
        w = nbr[1]
        nw = nbr[2]

        if pad_img[n] == 0 and pad_img[w] == 0 and pad_img[nw] == 0:
          # print('all zero neighbors')
          pad_img[idx] = label
          parent.append(0)
          label += 1

        elif pad_img[nw] != 0: #pixel will use nw label regardless of other neighbors
          # print('nw label')
          # print(labeled_image[nw])
          pad_img[idx] = pad_img[nw]

          # if pad_img[n] != pad_img[w] and \
          # (pad_img[n] and pad_img[w]) != 0:
          #   union(pad_img[n], pad_img[w], parent)
          #   union(pad_img[nw], pad_img[n], parent)

          # elif pad_img[nw] != pad_img[n] and \
          # pad_img[n] != 0:
          #   union(pad_img[nw], pad_img[n], parent)

          # elif pad_img[nw] != pad_img[w] and \
          # pad_img[w] != 0:
          #   union(pad_img[nw], pad_img[w], parent)

        elif pad_img[nw] == 0 and pad_img[n] == 0 and pad_img[w] != 0:
          # print('w label')
          pad_img[idx] = pad_img[w]

        elif pad_img[nw] == 0 and pad_img[w] == 0 and pad_img[n] != 0:
          # print('n label')
          pad_img[idx] = pad_img[n]

        # elif labeled_image[n] != labeled_image[w]:
        #   print('n and w not equal')
        elif pad_img[nw] == 0 and pad_img[w] != 0 and pad_img[n] != 0:
          if pad_img[w] != pad_img[n]:
            pad_img[idx] = min(pad_img[n], pad_img[w])
            union(pad_img[n], pad_img[w], parent)
          else:
            pad_img[idx] = pad_img[n]

        else:
          pad_img[idx] = min(pad_img[n], pad_img[w])

          if pad_img[n] != pad_img[w]:
            union(pad_img[n], pad_img[w], parent)

  # print(np.amax(pad_img))
  ## Pass 2
  for r in range(0, pad_img.shape[0]):
    for c in range(0, pad_img.shape[1]):
      if pad_img[r, c] != 0:
        pad_img[r, c] = find(pad_img[r, c], parent)       

  # ## Calculate new gray labels for better visualization
  # unique = np.unique(pad_img)
  # # print(unique)
  # lo = 0
  # hi = 200
  # gray_labels = np.linspace(lo, hi, len(unique)).astype(int)
  # print(gray_labels)

  # # make sure that gray_labels are different from original unique labels
  # for gray in range(1, len(gray_labels)): # but skip the first 0 (background)
  #   if gray_labels[gray] in unique:
  #     gray_labels[gray] += 1
  
  # for label in range(1, len(gray_labels)):
  #   pad_img[pad_img == unique[label]] = gray_labels[label]

  
  labeled_image = pad_img[1:-1, 1:-1]
  # print(np.unique(labeled_image))
  # print(labeled_image.shape)

  return labeled_image

def union(x, y, parent):
  # if x and y don't have the same parent, then merge them by adding the larger label as a child of the smaller label
  # x and y are integer labels corresponding to indices of the parents list
  # parent is a list of parents for each label

  label1 = x
  label2 = y

  while parent[label1] != 0:
    label1 = parent[label1]
  while parent[label2] != 0:
    label2 = parent[label2]

  if label1 != label2:
    parent[max(label1, label2)] = min(label1, label2)

def find(x, parent):
  # finds the parent of x
  # x is an integer label corresponding to index of the parents list
  # parent is a list of parents for each label

  label = x

  while parent[label] != 0:
    label = parent[label]
  return label

def get_prior_neighbors(pixel):
  # returns a list of prior neighbors represented as tuples
  # x is a tuple describing pixel coordinates

  prior_neighbors = []
  n = (pixel[0]-1, pixel[1])
  if n[0] >= 0:
    prior_neighbors.append(n)

  w = (pixel[0], pixel[1]-1)
  if w[1] >= 0:
    prior_neighbors.append(w)

  nw = (pixel[0]-1, pixel[1]-1)
  if nw[0] >= 0 and nw[1] >= 0:
    prior_neighbors.append(nw)

  return prior_neighbors


def get_attribute(labeled_image):
  # TODO
  attribute_list = []

  labels, counts = np.unique(labeled_image, return_counts=True)
  labels = labels[1:] #remove the 0 for background
  counts = counts[1:]

  for obj in range(len(labels)):
    area = counts[obj]

    #calculate position
    x_sum = 0.
    y_sum = 0.

    # for r in range(0, labeled_image.shape[0]):
    #   for c in range(0, labeled_image.shape[1]):
    #     if labeled_image[r,c] == labels[obj]:
    #       x_sum += c
    #       y_sum += r

    idx = np.argwhere(labeled_image == labels[obj])
    x_sum = np.sum(idx[:,1])
    y_sum = np.sum(idx[:,0])

    x_pos = x_sum/area
    y_pos = y_sum/area
    y_pos = labeled_image.shape[0]-y_pos-1 # convert to x, y bottom left coords
    
    position = {'x':x_pos.item(), 'y':y_pos.item()}

    #calculate orientation
    a = 0.
    b = 0.
    c = 0.
    # for row in range(0, labeled_image.shape[0]):
    #   for col in range(0, labeled_image.shape[1]):
    #     if labeled_image[r,c] == labels[obj]:
    #       a += (c-x_pos)**2
    #       b += 2*(c-x_pos)*(labeled_image.shape[0]-1-r-y_pos)
    #       c += (labeled_image.shape[0]-1-r-y_pos)**2
    for i in idx:
      x = i[1]
      y = labeled_image.shape[0]-1 - i[0]

      a += (x-x_pos)**2
      b += 2*(x-x_pos)*(y-y_pos)
      c += (y-y_pos)**2

    theta_min = 0.5*np.arctan2(b, a-c)

    # adjust quandrants so angles are within range of 0 to pi
    if theta_min >= -np.pi and theta_min < 0: # quadrants 3 and 4
      theta_min += np.pi

    if theta_min <= np.pi/2:
      theta_max = theta_min + np.pi/2
    else:
      theta_max = theta_min - np.pi/2

    orientation = theta_min

    # calculate roundness
    E_min = a*(np.sin(theta_min))**2 - b*np.sin(theta_min)*np.cos(theta_min) +\
    c*(np.cos(theta_min))**2
    E_max = a*(np.sin(theta_max))**2 - b*np.sin(theta_max)*np.cos(theta_max) +\
    c*(np.cos(theta_max))**2

    roundness = E_min/E_max

    # create dictionary and add to attribute list
    obj_dict = {'position':position, 'orientation':orientation.item(), 'roundedness':roundness.item()}
    attribute_list.append(obj_dict)

  # Create an image displaying center and orientation lines
  img_test = np.copy(labeled_image)

  for d in attribute_list:
    center = d['position']
    orientation = d['orientation']
    # print(type(orientation))

    # p1 = (int(center['x']) + int(50*np.cos(orientation)), \
    #   img_test.shape[0] - int(center['y'])-1 - int(50*np.sin(orientation)))
    # p2 = (int(center['x']) - int(50*np.cos(orientation)), \
    #   img_test.shape[0] - int(center['y'])-1 + int(50*np.sin(orientation)))

    cv2.circle(img_test, (int(center['x']), img_test.shape[0] - int(center['y']) -1), 1, 255, -1)
    # cv2.line(img_test, p1, p2, 255, 2)
  
  cv2.imwrite('output/' + "img_attribute_test.png", img_test)

  return attribute_list

def main(argv):
  img_name = argv[0]
  thresh_val = int(argv[1])

  img = cv2.imread('data/' + img_name + '.png', cv2.IMREAD_COLOR)
  gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  binary_image = binarize(gray_image, thresh_val=thresh_val)
  labeled_image = label(binary_image)
  attribute_list = get_attribute(labeled_image)

  center_img = np.copy(img)
  for att in attribute_list:
    center = att['position']
    cv2.circle(center_img, (int(center['x']), center_img.shape[0] - int(center['y']) -1), 1, (0,0,255), -1)

  cv2.imwrite('output/' + img_name + "_gray.png", gray_image)
  cv2.imwrite('output/' + img_name + "_binary.png", binary_image)
  cv2.imwrite('output/' + img_name + "_labeled.png", labeled_image)
  cv2.imwrite('output/' + img_name + "_centers.png", center_img)


if __name__ == '__main__':
  main(sys.argv[1:])

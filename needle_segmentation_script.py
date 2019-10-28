from needle_segmentation_functions import *
import os
import cv2
import matplotlib.pyplot as plt

directory = ('C:\\Users\\dlezcan1\\Documents\\DL-Sharing-Folder'
               '\\Emily_covered_2019-10-17-11-10-21\\')
directory = ('C:\\Users\\dlezcan1\\Documents\\DL-Sharing-Folder\\'
             '10-28-19_New-Images\\')
os.chdir(directory)

for file in os.listdir():
    if file.endswith('.png') and not '_ROI' in file and False:
        print("Processing file {}...".format(file))
        cv2.imshow("thresh_"+file,segment_needle(file,"thresh"))
        print("{} processed.\n".format(file))
        break;
        
cv2.waitKey(0)
cv2.destroyAllWindows()

# iterate through file directory
for file in os.listdir():
    if (file.endswith('.png') and not '_ROI' in file
        and not '_processed' in file and True):
        print("Processing file {}...".format(file))
        roi_image, seg_needle = segment_needle(file,"canny",False)
        cv2.imshow("Skeletonized canny:"+file,seg_needle)
        print("{} processed.\n".format(file))
        seg_needle[ seg_needle > 0] = 1
        roi_filename = directory + file[:-4] + '_ROI.png'
        cv2.imwrite(roi_filename, roi_image)
        print("File written:", roi_filename)
        x, y, p = fit_polynomial_skeleton(seg_needle,12)
        plt.figure()
        
        
        plt.gray()
        roi_image = plt.imread(roi_filename)
        plt.imshow(roi_image)
        plt.plot( x, p(x), 'r-') # ,x,y, 'b.',)
##        plt.xlim(0,seg_needle.shape[1])
##        plt.ylim(0,seg_needle.shape[0])
        plt.title(file)
        manager = plt.get_current_fig_manager()
        manager.resize(*manager.window.maxsize())
        save_filename = directory + file[:-4] + '_processed.png'
##        plt.savefig(save_filename, format='png', dpi=500)
        print('Figure saved:', save_filename)
##        break

# pick out a particular file
if False:
    file = directory + 'image2.png'
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = img[ROI[2]:ROI[3],ROI[0]:ROI[1]]
    segment_needle(file,'canny', True)
    find_coordinate_image(img)
        

plt.show()
cv2.waitKey(0)
plt.close('all')
cv2.destroyAllWindows()

from needle_segmentation_functions import *
import os
import cv2
import matplotlib.pyplot as plt
import image_processing as imgp

directory = "Test Images/Curvature_Experiment_11-15-19/"
directory = "../FBG_Needle_Calibration_Data/needle_1/"
print( "CWD:", os.getcwd() )

for file in os.listdir():
    if file.endswith( '.png' ) and not '_ROI' in file and False:
        print( "Processing file {}...".format( file ) )
        cv2.imshow( "thresh_" + file, segment_needle( file, "thresh" ) )
        print( "{} processed.\n".format( file ) )
        break;
        
cv2.waitKey( 0 )
cv2.destroyAllWindows()

# iterate through file directory
for file in os.listdir( directory ):
    if ( file.endswith( '.png' ) and not '_ROI' in file
        and not '_processed' in file and False ):
        full_name = directory + file
        print( "Processing file {}...".format( full_name ) )
        
        roi_image, seg_needle = segment_needle( full_name, "canny", False )
        
        cv2.imshow( "Skeletonized canny:" + file, seg_needle )
        print( "{} processed.\n".format( file ) )
        seg_needle[ seg_needle > 0] = 1
        roi_filename = "Output/" + file[:-4] + '_ROI.png'
        cv2.imwrite( roi_filename, roi_image )
        print( "File written:", roi_filename )
        x, y, p = fit_polynomial_skeleton( seg_needle, 12 )
        plt.figure()
        
        plt.gray()
        roi_image = plt.imread( roi_filename )
        plt.imshow( roi_image )
        plt.plot( x, p( x ), 'r-' )  # ,x,y, 'b.',)
# #        plt.xlim(0,seg_needle.shape[1])
# #        plt.ylim(0,seg_needle.shape[0])
        plt.title( file )
        manager = plt.get_current_fig_manager()
        manager.resize( *manager.window.maxsize() )
        save_filename = directory + file[:-4] + '_processed.png'
# #        plt.savefig(save_filename, format='png', dpi=500)
        print( 'Figure saved:', save_filename )
# #        break

# pick out a particular file
if True:
    file = directory + "12-19-19_12-32/mono_0019.jpg"
    img = cv2.imread( file, cv2.IMREAD_GRAYSCALE )
    ROI = [84, 250, 1280, 715]  # x_t-left, y_t-left, x_b-right, y_b-right
#     img = imgp.set_ROI(img, crop_area)
    CROP_AREA = ( 32, 425, 1180, 580 )
    img = imgp.set_ROI_box(img, CROP_AREA)
    find_coordinate_image( img )
#     seg_needle, _ = segment_needle(file,'canny', True)

plt.show()
cv2.waitKey( 0 )
plt.close( 'all' )
cv2.destroyAllWindows()

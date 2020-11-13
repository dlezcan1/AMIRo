# AMIRo
This repository is to support FBG-based needle shape sensing projects and needle segmentation tasks. 

## FBG Signal Processing
Packages here include automation of processing needle calibration data and performing the calibration. 
The caliibration method currently uses a simple least squares method. 

### Code for calibration:
* `calibrationMatrix.py`
* `FBGNeedle.py`: class file for FBG-sensorized needle wrapper
* `fbgCalibration.py`

### Code for FBG signal processing
* `hyperion.py`: hyperion si155 interrogator python interface. 
* `async_getFBGPeaks.py`
* `basic_getFBGPeaks.py`
* `findHyperionIP.py`
* several bash scripts in `src/bash_scripts/`

## Needle segmentation
Code here is used for needle segmentation tasks for ground truth image reconstruction of needles insertions.

### Code
* `image_processing.py`
* `needle_segmentation_functions.py`
* `needle_segmentation_scipt.py`
* `needle_segmentation_test.py`
* `stereo_needle_proc.py`: stereo image processing for needle segmentation and 3-D reconstruction

## Data
There are several images including stereo needle insertions and needle calibration monocular images used for needle segmentation tasks. 
FBG signal data here is from FBG-interlaced needles from calibrations performed using jig and 2-D needle segmentations.

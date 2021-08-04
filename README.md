# AMIRo
This repository is to support FBG-based needle shape sensing projects and needle segmentation tasks. 

## FBG Signal Processing
Packages here include automation of processing needle calibration data and performing the calibration. 
The caliibration method currently uses a simple least squares method. 

### Code for calibration:
* `fbg_needle_calibration.py`: script for calibrating FBGNeedles
* `sensorized_needles.py`: class file for FBG-sensorized needle wrapper
* `open_files.py`: library for reading in experimental data files
* `fbg_signal_processing.py`: library of FBG signal processing methods 

### `hyperion_interface` Code for FBG interface 
* `hyperion.py`: hyperion si155 interrogator python interface library.
* `async_getFBGPeaks.py`
* `basic_getFBGPeaks.py`
* `getFBGPeaks.py`
* `findHyperionIP.py`
* `plotFBGPeaks.py`: script to see the spectrum ov the interrogator
* several bash scripts in `src/bash_scripts/` for operating these scripts during experiment

### `matlab_scripts` MATLAB Functions for shape sensing and data analysis
This is where analysis in MATLAB is performed. Current version is for MATLAB R2021a

### Other Repositories
* `shape_sensing`: For FBG needle shape sensing
* `amiro-cv`: for computer vision applications of needle shape sensing experiments

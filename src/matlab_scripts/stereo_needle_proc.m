%% stereo_needle_proc.m
%
% Perform needle 3d segmentation (MATLAB wrapper for Python code written)
%
% - written by: Dimitri Lezcano
clear all;

mod = py.importlib.import_module('stereo_needle_proc');
py.importlib.reload(mod);

%% Set-up 
% options

% python set-up
if ispc % windows file system
    pydir = "..\";
    
else
    pydir = "../";
    
end

if count(py.sys.path, pydir) == 0
    insert(py.sys.path, int32(0), pydir);
end

None = py.None;

% directories for files
stereo_param_dir = '../../Stereo_Camera_Calibration_10-23-2020/';
stereo_needle_dir = '../../Test Images/stereo_needle/needle_examples/';
stereo_param_cvfile = stereo_param_dir + "calibrationSession_params-error_opencv-struct.mat";
stereo_param_file = stereo_param_dir + "calibrationSession_params-error.mat";

% load the stereo parameters
stereo_params = load(stereo_param_file, 'stereoParams').stereoParams;
stereo_params_py = py.stereo_needle_proc.load_stereoparams_matlab(stereo_param_cvfile);

%% Stereo processing
% read in the images
num = 6;
file_base = "%s-%04d.png";

l_img_file = stereo_needle_dir + sprintf(file_base, 'left', num);
r_img_file = stereo_needle_dir + sprintf(file_base, 'right', num);

l_img = imread(l_img_file);
r_img = imread(r_img_file);

needle_proc(l_img, r_img)

% stereo needle processing
roi_l = py.tuple({{int16(70), int16(80)}, {int16(500), int16(915)}});
roi_r = py.tuple({{int16(70), int16(55)}, {int16(500), int16(-1)}}); 
res = py.stereo_needle_proc.needleproc_stereo(py.numpy.array(l_img), py.numpy.array(r_img),...
                                              py.list(), py.list(), roi_l, roi_r);                
left_skel = boolean(res{1});
right_skel = boolean(res{2});
conts_l = res{3};
conts_r = res{4};

% perform contour matching
res = py.stereo_needle_proc.stereomatch_needle(conts_l{1}, conts_r{1});
cont_l_match = squeeze(double(res{1}));
cont_r_match = squeeze(double(res{2}));

%% 3-D Reconstruction
needle_3d = triangulate_stereomatch(cont_l_match, cont_r_match, stereo_params);

plot3(needle_3d(:,1), needle_3d(:,2), needle_3d(:,3));
axis equal; grid on;




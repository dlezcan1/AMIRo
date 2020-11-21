%% opencv_stereoparams.m
%
% Function to convert matlab stereoparams to opencv format of json file.
%
% - written by: Dimitri Lezcano

function [S, save_file] = opencv_stereoparams(file, out_file)
    
    % load the file
    m = load(file);
    
    % get the stereo params
    stereoParams = m.stereoParams;
    
    % struct for data processing
    S.camera1 = struct();
    S.camera2 = struct();
    
    % camera intrinsics
    S.camera1.intrinsics.cameramatrix = stereoParams.CameraParameters1.IntrinsicMatrix';
    S.cameraMatrix1 = S.camera1.intrinsics.cameramatrix;
    S.camera2.intrinsics.cameramatrix = stereoParams.CameraParameters2.IntrinsicMatrix';
    S.cameraMatrix2 = S.camera2.intrinsics.cameramatrix;
    
    % distortion coefficients
    if length(stereoParams.CameraParameters1.RadialDistortion) == 2
        S.camera1.intrinsics.distCoeffs = [stereoParams.CameraParameters1.RadialDistortion(1:2), ...
                                           stereoParams.CameraParameters1.TangentialDistortion, ...
                                           0];
    else
        S.camera1.intrinsics.distCoeffs = [stereoParams.CameraParameters1.RadialDistortion(1:2), ...
                                           stereoParams.CameraParameters1.TangentialDistortion, ...
                                           stereoParams.CameraParameters1.RadialDistortion(3)];
    end
    S.distCoeffs1 = S.camera1.intrinsics.distCoeffs;
    
    if length(stereoParams.CameraParameters1.RadialDistortion) == 2
        S.camera2.intrinsics.distCoeffs = [stereoParams.CameraParameters2.RadialDistortion(1:2), ...
                                           stereoParams.CameraParameters2.TangentialDistortion, ... 
                                           0];
    else
        S.camera2.intrinsics.distCoeffs = [stereoParams.CameraParameters2.RadialDistortion(1:2), ...
                                           stereoParams.CameraParameters2.TangentialDistortion, ...
                                           stereoParams.CameraParameters2.RadialDistortion(3)];
    end
    S.distCoeffs2 = S.camera2.intrinsics.distCoeffs;
    
    % camera extrinsics
    S.camera1.extrinsics.R = stereoParams.CameraParameters1.RotationMatrices;
    S.R1 = S.camera1.extrinsics.R;
    S.camera1.extrinsics.tvecs = stereoParams.CameraParameters1.TranslationVectors;
    S.tvecs1 = S.camera1.extrinsics.tvecs;
    
    S.camera2.extrinsics.R = stereoParams.CameraParameters2.RotationMatrices;
    S.R2 = S.camera2.extrinsics.R;
    S.camera2.extrinsics.tvecs = stereoParams.CameraParameters2.TranslationVectors;
    S.tvecs2 = S.camera2.extrinsics.tvecs;
    
    % stereo extrinics/geometry
    S.R = stereoParams.RotationOfCamera2';
    S.t = stereoParams.TranslationOfCamera2';
    S.F = stereoParams.FundamentalMatrix';
    S.E = stereoParams.EssentialMatrix';
    S.units = stereoParams.WorldUnits;
    
    % save to json
    if out_file
        save_file = strrep(file, '.mat', '_opencv-struct.mat');
        save(save_file, '-struct', 'S');
        fprintf('Saved struct to file: %s\n\n', save_file);
    end


end
%% align_camera_needle_shapes.m
% - written by: Dimitri Lezcano
%
% this is a function to align the camera shape to the needle coordinates
%
% Args:
%   - shape_cam:  the camera shape N x 3 coordinates
%   - shape_ndl:  the needle shape M x 3 coordinates
%   - aruco_pose: the aruco pose in needle coordinates (from calibration)
%   - options:
%       - interpolate: Whether to interpolate the coordinates or not
%                         (Default = true)
%       - sCutoff:     [s_low, s_high] values for arclengths to include in 
%                         fitting (Default = [0 inf])
%       - ds:          the interpolation arclength increment (Default = 0.5)
%       - RotateZ:     Whether to determine optimal z-rotation to align the
%                        frames (to account for any offset rotations from
%                        experiment) (Default = true)
%
% Return:
%   - transformed (and interpolated) camera shape into needle coordinates
%   - (interpolated) needle shape
%   - 4x4 Transformation matrix from camera -> needle
%   - theta_z: the optimal rotation z-axis rotation amount to align needle
%               ( this is to offset any rotations not recorded in expmt.)

%% Function
function [cam_tf, ndl, varargout] = align_camera_needle_shapes(shape_cam, shape_ndl,...
    cam_aruco_pose, needle_aruco_pose_cal, options)
    arguments
        shape_cam             (:,3);
        shape_ndl             (:,3);
        cam_aruco_pose        (4,4); % T_CA
        needle_aruco_pose_cal (4,4); % T_FA
        options.interpolate   logical = true;
        options.sCutoff       (1,2)   = [0 inf];
        options.ds            double  = 0.5;
        options.RotateZ       logical = true;
    end
    %% Argument checking
    assert(options.interpolate); % haven't implemented non-interpolation
    
    %% Determine the arclengths of the shapes
    [arclen_cam, ~, s_cam] = arclength(shape_cam);
    [arclen_ndl, ~, s_ndl] = arclength(shape_ndl);
        
    %% Interpolate the shapes
    if options.interpolate
       % determine interpolation arclengths
       s_cam_interp = flip(arclen_cam:-options.ds:0);
       s_ndl_interp = 0:options.ds:arclen_ndl;
       
       % interpolate the shapes
       cam = interp_pts(shape_cam, s_cam_interp);
       ndl = interp_pts(shape_ndl, s_ndl_interp);
       
       % update the arclength parametrizations
       s_cam = s_cam_interp;
       s_ndl = s_ndl_interp;
       
    else
       cam = shape_cam;
       ndl = shape_ndl;
       
    end
    
    
    %% Rotate camera shape to needle coordinates
    needle_cam_pose = needle_aruco_pose_cal * finv(cam_aruco_pose);
    cam_tf_homo = [cam, ones(size(cam,1),1)] * needle_cam_pose';
    cam_tf = cam_tf_homo(:,1:3); % remove last column
    
    if options.RotateZ
        % arclengths to keep
        mask_cam_fit = options.sCutoff(1) <= s_cam & s_cam <= options.sCutoff(2);
        mask_ndl_fit = options.sCutoff(1) <= s_ndl & s_ndl <= options.sCutoff(2);
        
        % grab the needle points to fit
        ndl_fit    =    ndl(mask_ndl_fit, 1:2);
        cam_tf_fit = cam_tf(mask_cam_fit, 1:2);
        N = min(size(ndl_fit, 1), size(cam_tf_fit, 1));
        
        % Find the optimal Z-rotation 
        R2d            = fit_rot2d(ndl_fit(1:N,:)', cam_tf_fit(1:N,:)' );
        thetaz         = theta_rot2d(R2d); %atan2(R2d(2,1), R2d(1,1));
        pose_R2d       = [rotz(thetaz), zeros(3,1); 
                          zeros(1,3),   1];
        % adjust needle pose
        pose_ndl_cam = pose_R2d * needle_cam_pose; % T_NC
        cam_tf       = cam_tf * rotz(thetaz)';
        
    else
        thetaz       = 0;
        pose_ndl_cam = needle_cam_pose;
        
    end
    
    % output coordinates
    varargout{1} = pose_ndl_cam;
    varargout{2} = thetaz;
    
end

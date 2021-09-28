%% read_aruco_pose.m
%
% this is a function to return the ARUCO pose
% 
% Args:
%   - pose_file: string of the ARUCO pose file 
%
% Returns:
%    - 4x4 left pose matrix 
%    - 4x4 right pose matrix
%    - ARUCO ID
function [pose_l, pose_r, id] = read_aruco_pose(pose_file)
    arguments
        pose_file string;
    end
    
    id_pose = readmatrix(pose_file, 'Delimiter', {',', ':'});
    
    id = id_pose(1);
    
    % left pose
    rvec_l = id_pose(2:4);
    tvec_l = reshape(id_pose(5:7), 3, 1);
    
    if all(~isnan([rvec_l, tvec_l']))
        R_l = rotationVectorToMatrix(rvec_l);
        pose_l = [R_l tvec_l; zeros(1,3), 1];
    else
        pose_l = nan;
    end
    
    % right pose
    rvec_r = id_pose(8:10);
    tvec_r = reshape(id_pose(11:13), 3, 1);
    
    if all(~isnan([rvec_r, tvec_r']))
        R_r = rotationVectorToMatrix(rvec_r);
        pose_r = [R_r tvec_r; zeros(1,3), 1];
    else
        pose_r = nan;
    end
    
end

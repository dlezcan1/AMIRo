%% point_cloud_reg_tip.m
%
% function to perform point-cloud registration using SVD with tip
%   alignment
%
% Args:
%   - a, b: the two different 3-D point clouds to register to each other
%           of size [ N x 3 ]
%   - method (Default: 'horn'): method for computing 3D registration
%
% Return:
%       - R, p: rotation matrix, transaltion vector s.t.
%           R * a + p = b
% 
% - written by: Dimitri Lezcano

function [R, p] = point_cloud_reg_tip(a, b, method)
    arguments
        a (:,3);
        b (:,3);
        method string = 'horn'
    end
    
    %% coincide the tips
    a_hat = a - a(end, :); 
    b_hat = b - b(end, :);
    
    %% Get the rotaiton matrix
    % determine method
    if strcmp(method, "horn") == 1
        R = determine_rotation_horn(a_hat, b_hat);
        
    else
        error("Method '%s' is not implemented.", method);
        
    end
    
    %% Get the translation
    p = b(end, :)' - R * a(end, :)';
    
end

%% helper functions
% determine the rotation using horn's method
function R = determine_rotation_horn(a_hat, b_hat)
    % taken from CIS 1 homework 
    
    % set-up
    H = a_hat'*b_hat; % compute H matric
    delta = vee(H' - H);
    
    % compose G matric
    G = [ trace(H), delta'; 
          delta, H + H' - trace(H)*eye(3)];
      
    % perform eigenvalue decomposition of G
    [V, D] = eig(G);
    
    eigvals = diag(D);
    [~, maxidx] = max(eigvals);
    quat = V(:, maxidx); % unit quaternion
    quat = quat' / norm(quat); % ensure unit
%     quat = reshape(quat, 1, 4);
    
    % convert quaternion to rotation matrix
    R = quat2rotm(quat);    
        
end
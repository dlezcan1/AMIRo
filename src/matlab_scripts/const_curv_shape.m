%% const_curv_shape.m
% jig (constant curvature) shape model
function r = const_curv_shape(w, s)
% function to get the constant curvature shape t
%
% constant curvature is 1/k * v | k = curvature, v = "torque" vector
% 
% Input:
%   - w: the angular deformation vector (constant)
%   - s: the arclength coordinates N-vector

    wv = w .* ones(3, length(s));
    
    r = wv2r(wv, max(s));
    
end
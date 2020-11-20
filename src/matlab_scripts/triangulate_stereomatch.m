%% triangulate_stereomatch.m
%
% this is a function to perform the triangulation of the needle using a
% stereomatch 
%
% - written by: Dimitri Lezcano

function points_3d = triangulate_stereomatch(left_match, right_match, stereo_params)
    points_3d = simple_triangulation(left_match, right_match, stereo_params);
    
end

% simple triangulation
function points_3d = simple_triangulation(left_match, right_match, stereo_params)
    % simple triangulation
    points_3d = triangulate(left_match, right_match, stereo_params);
    
end
%% plot_tissueboundary3d.m
% - written by: Dimitri Lezcano
%
% this is a function to plot the Tissue boundary in needle coordinates for
% a 3D plot
%
% Args:
%   - z_crit: the length of the tissue layer
%   - xc:     the center x-coordinate of point to plot the axis
%   - yc:     the center y-coordinate of point to plot the axis
%   - width:  (Default = 5) the width  of the tissue layer plot
%   - height: (Default = 5) the height of the tissue layer plot
%
%

function plot_tissueboundary3d(z_crit, xc, yc, width, height)
    arguments
        z_crit double;
        xc double;
        yc double;
        width double = 5;
        height double = 5;
    end
    
   S1 = [z_crit, z_crit;
         xc - width, xc + width;
         yc - height, yc + height];
   S2 = S1;
   S2(2,:) = S1(2,end:-1:1);
   S = [S1(:,1) S2(:,1) S1(:,2) S2(:,2)];
   patch(S(1,:), S(2,:), S(3,:), 'r', 'DisplayName', 'Tissue Boundary');
         
end
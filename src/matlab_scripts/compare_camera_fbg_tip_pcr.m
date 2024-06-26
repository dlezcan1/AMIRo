%% compare_camera_fbg_tip_pcr.m
% 
% this is a script to compare shape sensing methods to FBG shape sensing
% using the tip-aligned point-cloud registration
%
% - written by: Dimitri Lezcano
clear; close all;
%% Set-Up
% directories to iterate throughn ( the inidividual trials )
expmt_dir = "../../data/3CH-4AA-0004/2021-10-04_Insertion-Expmt-1/"; % CAN CHANGE
trial_dirs = dir(fullfile(expmt_dir, "Insertion*/"));
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories and "0" directory
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% stereo parameters
stereoparam_dir = "../../amiro-cv/calibration/Stereo_Camera_Calibration_02-08-2021/6x7_5mm/";
stereoparam_file = fullfile(stereoparam_dir, "calibrationSession_params-error.mat");
stereoParams = load(stereoparam_file).stereoParams;

% FBG reliability weight options
use_weights = true; % CAN CHANGE but USUALLY KEEP

% number of layers (CAN CHANGE)
num_layers = 1;    % implemented 1 or 2
singlebend = true; % doublebend if false

% saving options
save_bool = true; % CAN CHANGE
fileout_base = "%s_%s_FBG-Camera-Comp_tip-pcr";
if singlebend
    fileout_base = sprintf(fileout_base, 'SingleBend', '%s');
else
    fileout_base = sprintf(fileout_base, 'DoubleBend', '%s');
end
    
if num_layers == 1 
    fileout_base = sprintf(fileout_base, 'SingleLayer');
elseif num_layers == 2
    fileout_base = sprintf(fileout_base, 'DoubleLayer');
else
    error("Not Implemented");
end

if use_weights
    fileout_base = strcat(fileout_base, "_FBG-weights");
end

% 3D point file names
camera_pos_file = "left-right_3d-pts.csv";
fbg_pos_file = "FBGdata_%s3d-position.xls";
if use_weights
    fbg_pos_file = sprintf(fbg_pos_file, '%sFBG-weights_');
end

if num_layers == 1 && singlebend
    fbg_pos_file = sprintf(fbg_pos_file, '');
elseif num_layers == 2 && singlebend
    fbg_pos_file = sprintf(fbg_pos_file, '2layer_');
elseif num_layers == 1 && ~singlebend
    fbg_pos_file = sprintf(fbg_pos_file, 'doublebend_');
else
    error('Not Implemented');
end

% arclength options
ds = 0.5;

%% Read the experiment.json file
experiment_description = jsondecode(fileread(fullfile(expmt_dir, 'experiment.json')));
if ~singlebend
    s_dbl_bend = experiment_description.DoubleBendDepth;
elseif num_layers == 2
    z_crit = experiment_description.tissue1Length;
    s_dbl_bend = -1;
end

%% Create the base table
% setup the columns
col_names = {'Ins_Hole', 'L', 'singlebend','cam_shape', 'fbg_shape', 'L_cam', 'L_fbg',...
             'Pose_nc', 'RMSE', 'MaxError', 'MeanInPlane', 'MaxInPlane', ...
             'MeanOutPlane', 'MaxOutPlane'};
Ncols_err = numel(col_names) - find(strcmp(col_names, "RMSE")) + 1;
col_types = cat(2, {'uint8', 'double', 'logical','double', 'double', 'double', 'double', 'double'},...
                repmat({'double'}, 1, Ncols_err));
col_units = cat(2, {'', 'mm', '','mm', 'mm', 'mm', 'mm' ,'mm'},...
                repmat({'mm'}, 1, Ncols_err));
            
% create the empty table
fbg_cam_compare_tbl = table('Size', [0, numel(col_names)],...
                            'VariableTypes', col_types,...
                            'VariableNames', col_names);
fbg_cam_compare_tbl.Properties.VariableUnits = col_units;


%% Process each trial
progressbar('Processing Data...');
for i = 1:length(trial_dirs)
    progressbar(i/numel(trial_dirs));
    % trial determination
    L = str2double(trial_dirs(i).name);

    if singlebend
        doublebend = false;

    elseif L == s_dbl_bend+1
        L = s_dbl_bend;
        doublebend = true;
    
    elseif L > s_dbl_bend
        doublebend = true;

    else
        doublebend = false;
    
    end
    re_ret = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
    hole_num = str2double(re_ret{1}{1});
    
    % data files
    fbg_file    = fullfile(trial_dirs(i).folder, trial_dirs(i).name, fbg_pos_file);
    camera_file = fullfile(trial_dirs(i).folder, trial_dirs(i).name, camera_pos_file);
    
    % load the shapes
    if ~isfile(fbg_file) || ~isfile(camera_file)
        continue;
    end
        
    fbg_pos = readmatrix(fbg_file)';
    cam_pos = readmatrix(camera_file);
    cam_pos = cam_pos(:,1:3);
    
    % determine the arclengths
    arclen_fbg = arclength(fbg_pos);
    arclen_cam = arclength(cam_pos);
    
    % perform point-cloud registration
    [R_nc, p_nc, ...
     cam_pos_interp, s_cam,...
     fbg_pos_interp, s_fbg,...
     cam_pos_interp_tf, fbg_pos_interp_tf] = pcr_Cam2FBG(cam_pos, fbg_pos, ds, -1);
    F_nc = makeSE3(R_nc, p_nc);
    % error analysis
    N_overlap = min(size(cam_pos_interp_tf,1), size(fbg_pos_interp,1));
    errors = error_analysis(cam_pos_interp_tf(end-N_overlap+1:end,:),...
                            fbg_pos_interp(end-N_overlap+1:end,:));
    
    % append a row the the table
    fbg_cam_compare_tbl = [fbg_cam_compare_tbl;
                           {hole_num, L, ~doublebend, cam_pos_interp, fbg_pos_interp,...
                            arclen_cam, arclen_fbg, F_nc, ...
                            errors.RMSE, max(errors.L2), ...
                            mean(errors.in_plane), max(errors.in_plane),...
                            mean(errors.out_plane), max(errors.out_plane)}];
    
    
end
fbg_cam_compare_tbl = sortrows(fbg_cam_compare_tbl, [1,2]); % Sort by insertion hole and depth

%% Save the Data
if save_bool
    fileout_base_data = fullfile(expmt_dir, strcat(fileout_base, '_results'));
    
    save(strcat(fileout_base_data, '.mat'), 'fbg_cam_compare_tbl');
    fprintf("Saved results to: %s\n",strcat(fileout_base_data, '.mat')); 
    
    mask = ~contains(varfun(@class, fbg_cam_compare_tbl, 'OutputFormat', 'cell'), 'cell');
    writetable(fbg_cam_compare_tbl(:,mask), strcat(fileout_base_data, '.xlsx'));
    fprintf("Saved results to: %s\n",strcat(fileout_base_data, '.xlsx'));
end

%% Set-Up Plotting
% Figure sizing setup
fig_counter = 1;
figsize = [0.3, 0.4];
figinc = figsize + [0.0, 0.075];
num_rowscols = round(1./figsize);
lb_offset = [.05, 1 - 0.08 - figsize(2)];

% Figure instantiation and positioning
fig_fbg_2d = figure(fig_counter); fig_counter = fig_counter + 1;
col = mod(fig_counter - 2, num_rowscols(2));
row = floor((fig_counter - 2)/num_rowscols(1));
set(fig_fbg_2d, 'Units', 'Normalized', 'Position', ...
    [lb_offset + [col, -row].*figinc, figsize]);

fig_fbg_3d = figure(fig_counter); fig_counter = fig_counter + 1;
col = mod(fig_counter - 2, num_rowscols(2));
row = floor((fig_counter - 2)/num_rowscols(1));
set(fig_fbg_3d, 'Units', 'Normalized', 'Position', ...
    [lb_offset + [col, -row].*figinc, figsize]);

fig_cam_3d = figure(fig_counter); fig_counter = fig_counter + 1;
col = mod(fig_counter - 2, num_rowscols(2));
row = floor((fig_counter - 2)/num_rowscols(1));
set(fig_cam_3d, 'Units', 'Normalized', 'Position', ...
    [lb_offset + [col, -row].*figinc, figsize]);

fig_err    = figure(fig_counter); fig_counter = fig_counter + 1;
col = mod(fig_counter - 2, num_rowscols(2));
row = floor((fig_counter - 2)/num_rowscols(1));
set(fig_err, 'Units', 'Normalized', 'Position', ...
    [lb_offset + [col, -row].*figinc, figsize.*[1.5,1]]);

fig_comp_2d = figure(fig_counter); fig_counter = fig_counter + 1;
col = mod(fig_counter - 2, num_rowscols(2));
row = floor((fig_counter -2)/num_rowscols(1));
set(fig_comp_2d, 'Units', 'Normalized', 'Position', ...
    [lb_offset + [col, -row].*[1.5, 1].*figinc, figsize.*[1.5,1]]);

fig_img_proj = figure(fig_counter); fig_counter = fig_counter + 1;
set(fig_img_proj, 'Units', 'Normalized', 'Position', ...
    [1 + 0.1, 0.1, 0.8, 0.8]); % on another screen

%% Plotting
for ins_hole = unique(fbg_cam_compare_tbl.Ins_Hole)'
    sub_tbl = fbg_cam_compare_tbl(fbg_cam_compare_tbl.Ins_Hole == ins_hole, :);
    fprintf("Plotting Insertion %d\n", ins_hole);
    
    for j = 1:size(sub_tbl,1)
       % trial results
       p_fbg = sub_tbl.fbg_shape{j}; 
       p_cam = sub_tbl.cam_shape{j};
       F_nc  = sub_tbl.Pose_nc{j};
       p_cam_tf = transformPointsSE3(p_cam, F_nc, 2); % camera shape in FBG coordinates
       p_fbg_tf = transformPointsSE3(p_fbg, finv(F_nc), 2); % FBG shape in camera coordinates
       L = sub_tbl.L(j);
       
       % trial-specific files
       trial_dir = fullfile(expmt_dir, sprintf("Insertion%d/%d", ins_hole, L));
       left_file = fullfile(trial_dir, 'left.png');
       right_file = fullfile(trial_dir, 'right.png');
       
       % FBG shape 2D view
       figure(fig_fbg_2d);
       subplot(2,1,1);
       if j == 1
           hold off;
       end
       plot(p_fbg(:,3), p_fbg(:,1), 'linewidth', 2); hold on;
       if j == 1 && num_layers == 2
           xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold on;
       elseif j == 1 && ~singlebend
           xline(s_dbl_bend, 'r--', 'Double Bend', 'DisplayName', 'Double Bend'); hold on;
       end
       axis equal; grid on;
       ylabel('x (mm)');
       
       subplot(2,1,2);
       if j == 1
           hold off;
       end
       plot(p_fbg(:,3), p_fbg(:,2), 'linewidth', 2); hold on;
       if j == 1 && num_layers == 2
           xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold on;
       elseif j == 1 && ~singlebend
           xline(s_dbl_bend, 'r--', 'Double Bend', 'DisplayName', 'Double Bend'); hold on;
       end
       axis equal; grid on;
       xlabel('z (mm)'); ylabel('y (mm)');
       
       sgtitle(sprintf("Insertion %d: FBG Shape", ins_hole));
       
       % FBG shape 3D view
       figure(fig_fbg_3d);
       if j == 1
           hold off;
       end
       plot3(p_fbg(:,3), p_fbg(:,1), p_fbg(:,2), 'linewidth', 2); hold on;
       if j == 1 && num_layers == 2
           plot_tissueboundary3d(z_crit, 0, 0, 5, 5);
       elseif j == 1 && ~singlebend
           plot_tissueboundary3d(s_dbl_bend, 0, 0, 5, 5);
       end
       axis equal; grid on;
       xlabel('z (mm)'); ylabel('x (mm)'); zlabel('y (mm)');
       title(sprintf("Insertion %d: FBG Shape", ins_hole));
       
       % stereo 3D view
       figure(fig_cam_3d);
       if j == 1
           hold off;
       end
       plot3(p_cam(:,1), p_cam(:,2), p_cam(:,3), 'linewidth', 2); hold on;
       axis equal; grid on;
       xlabel('x (mm)'); ylabel('y (mm)'); zlabel('z (mm)');
       title(sprintf("Insertion %d: Camera Shape", ins_hole));
       
       % FBG-Stereo 2D comparisons
       figure(fig_comp_2d);
       subplot(2,1,1);
       hold off;
       plot(p_fbg(:,3), p_fbg(:,1), 'r', 'linewidth', 2, 'DisplayName', 'FBG'); hold on;
       plot(p_cam_tf(:,3), p_cam_tf(:,1), 'g', 'linewidth', 2, 'DisplayName', 'stereo'); hold off;       
       if num_layers == 2
           xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold off;
       elseif ~singlebend
           xline(s_dbl_bend, 'r--', 'Double Bend', 'DisplayName', 'Double Bend'); hold on;
       end
       axis equal; grid on;
       legend('location', 'nw');
       ylabel('x (mm)');
       
       subplot(2,1,2);
       hold off;
       plot(p_fbg(:,3), p_fbg(:,2), 'r', 'linewidth', 2, 'DisplayName', 'FBG'); hold on;
       plot(p_cam_tf(:,3), p_cam_tf(:,2), 'g', 'linewidth', 2, 'DisplayName', 'stereo'); hold off;
       if num_layers == 2
           xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold off;
       elseif ~singlebend
           xline(s_dbl_bend, 'r--', 'Double Bend', 'DisplayName', 'Double Bend'); hold on;
       end
       axis equal; grid on;
       ylabel('y (mm)'); xlabel('z (mm)');
       
       sgtitle(sprintf("Insertion %d | Depth %d mm: FBG Shape", ins_hole, L));
       
       % Image projections
       figure(fig_img_proj);
       left_img = imread(left_file); right_img = imread(right_file);
       plotShapesInImage(left_img, right_img, p_cam, p_fbg_tf, stereoParams);
       title(sprintf("Insertion %d | Depth %d mm: Needle Shape Image Projections", ins_hole, L))
       
       % trial-based saving
       if save_bool
           fileout_base_j = fullfile(trial_dir, fileout_base);
           
           savefigas(fig_comp_2d, strcat(fileout_base_j, '_fbg-cam-2d'), ...
                        'Verbose', true);
                    
           savefigas(fig_img_proj, strcat(fileout_base_j, '_fbg-cam-img-proj'), ...
                        'Verbose', true);
           
       end
    end
    
    % Plot Errors
    figure(fig_err);
    subplot(1,3,1); 
    plot(sub_tbl.L, sub_tbl.RMSE, 'DisplayName', 'RMSE'); hold on;
    plot(sub_tbl.L, sub_tbl.MaxError, 'DisplayName', 'Max'); hold on;
    yline(0.5, 'r--'); hold off;
    erryl = [0, max([ylim, 1])];
    ylim(erryl)
    title("Distance Error"); ylabel('Error (mm)'); xlabel('Insertion Depth (mm)');
    legend({'RMSE', 'Max'}, 'Location', 'nw');
    grid on;
    
    subplot(1,3,2);
    plot(sub_tbl.L, sub_tbl.MeanInPlane, 'DisplayName', 'Mean'); hold on;
    plot(sub_tbl.L, sub_tbl.MaxInPlane, 'DisplayName', 'Max'); hold on;
    yline(0.5, 'r--'); hold off;
    ylim(erryl);
    title("In-Plane Error"); xlabel('Insertion Depth (mm)');
    legend({'Mean', 'Max'}, 'Location', 'nw');
    grid on;
    
    subplot(1,3,3);
    plot(sub_tbl.L, sub_tbl.MeanOutPlane, 'DisplayName', 'Mean'); hold on;
    plot(sub_tbl.L, sub_tbl.MaxOutPlane, 'DisplayName', 'Max'); hold on;
    yline(0.5, 'r--'); hold off;
    ylim(erryl);
    title("Out-of-Plane Error"); xlabel('Insertion Depth (mm)');
    legend({'Mean', 'Max'}, 'Location', 'nw');
    grid on;
    
    sgtitle(sprintf("Insertion %d: Shape Errors", ins_hole));
    
   % saving
   if save_bool
       fileout_base_j = fullfile(expmt_dir, sprintf("Insertion%d",ins_hole),...
                                 fileout_base);
                             
       savefigas(fig_err, strcat(fileout_base_j, '_fbg-cam-errors'), 'verbose', true);
       
       savefigas(fig_fbg_2d, strcat(fileout_base_j, '_fbg-shape-2d'), 'verbose', true);
       
       savefigas(fig_fbg_3d, strcat(fileout_base_j, '_fbg-shape-3d'), 'verbose', true);
       
       savefigas(fig_cam_3d, strcat(fileout_base_j, '_cam-shape-3d'), 'verbose', true);
   end
    
    disp(" ");
end

%% Summarize error plots
fig_err_sum = figure(fig_counter); fig_counter = fig_counter + 1;
pos_adj = [0, 0, 0, -0.075]; % adjust the height of the position
set(fig_err_sum, 'units', 'normalized', 'position', [0, 1/4, 8/9, 2/3]);
ax = subplot(1,3,1);
ax.Position = ax.Position + pos_adj;
boxplot(fbg_cam_compare_tbl.RMSE, fbg_cam_compare_tbl.L);
yl = [0, max([ylim, 1])]; ylim(yl); 
yline(0.5, 'r--');
xlabel('Insertion Depth (mm)'); ylabel('Error (mm)');
title('RMSE');

ax = subplot(1,3,2);
ax.Position = ax.Position + pos_adj;
boxplot(fbg_cam_compare_tbl.MeanInPlane, fbg_cam_compare_tbl.L);
ylim(yl); 
yline(0.5, 'r--');
xlabel('Insertion Depth (mm)'); ylabel('Error (mm)');
title('Mean IPE');

ax = subplot(1,3,3);
ax.Position = ax.Position + pos_adj;
boxplot(fbg_cam_compare_tbl.MeanOutPlane, fbg_cam_compare_tbl.L);
ylim(yl); 
yline(0.5, 'r--');
xlabel('Insertion Depth (mm)'); ylabel('Error (mm)');
title('Mean OPE');

expmt_dir_split = split(strip(fullfile(expmt_dir), filesep), filesep);
sgtitle(strrep(expmt_dir_split(end), '_', '-'));

if save_bool
    fileout_fig_err = fullfile(expmt_dir, strcat(fileout_base, '_summary-errors-windowed'));
    
    savefigas(fig_err_sum, fileout_fig_err, 'Verbose', true);
    
end

%% End Program
close all;


%% Helper functions
% interpolate the shape to constant arclength
function [shape_interp, s_interp] = interpolate_shape(shape, ds, flip_s)
    arguments
        shape (:,3);
        ds {mustBePositive} = 0.5;
        flip_s logical = false; % flip the arclength generation from the needle shapes
    end
    
    % determine arclengths
    arclen = arclength(shape);
    
    % generate interpolation arclengths
    if flip_s
        s_interp = flip(arclen:-ds:0);
    else
        s_interp = 0:ds:arclen;
    end
    
    % interpolate the needle shapes
    shape_interp = interp_pts(shape, s_interp);
    
    
end

% register the camera shape to the FBG shape
function [R_nc, p_nc, varargout] = pcr_Cam2FBG(p_cam, p_fbg, ds, min_s)
    arguments
        p_cam (:,3);
        p_fbg (:,3);
        ds {mustBePositive} = 0.5;
        min_s double = -1;
    end
    
    % interpolate the points
    [p_cam_interp, s_cam_interp] = interpolate_shape(p_cam, ds, true);
    [p_fbg_interp, s_fbg_interp] = interpolate_shape(p_fbg, ds, true);
    
    % ensure sorted arclengths
    [s_cam_interp, cam_idxs] = sort(s_cam_interp);
    p_cam_interp = p_cam_interp(cam_idxs, :);
    [s_fbg_interp, fbg_idxs] = sort(s_fbg_interp);
    p_fbg_interp = p_fbg_interp(fbg_idxs, :);
    
    % Determine the matching points
    N_match = min(sum(s_fbg_interp > min_s), sum(s_cam_interp > min_s));
    
    [R_nc, p_nc] = point_cloud_reg_tip( p_cam_interp(end - N_match + 1:end,:),...
                                        p_fbg_interp(end - N_match + 1:end,:));
    F_nc = makeSE3(R_nc, p_nc);
                                    
    % variable output arguments
    % - first interpolated
    varargout{1} = p_cam_interp;
    varargout{2} = s_cam_interp;
    varargout{3} = p_fbg_interp;
    varargout{4} = s_fbg_interp;
    % - Camera needle shape in needle frame
    varargout{5} = transformPointsSE3(p_cam_interp, F_nc, 2); 
    % - FBG needle shape in camera frame
    varargout{6} = transformPointsSE3(p_fbg_interp, finv(F_nc), 2); 
    
end


% error analysis
function errors = error_analysis(cam, fbg)
% measures error metrics from each points
    
    % L2 distance
    errors.L2 = vecnorm(cam - fbg, 2, 2); 
    
    % component-wise error
    errors.dx = abs(cam(:,1) - fbg(:,1));
    errors.dy = abs(cam(:,2) - fbg(:,2));
    errors.dz = abs(cam(:,3) - fbg(:,3));
    
    % in/out-plane error (assume in-plane is yz and out-plane is xz)
    errors.in_plane = vecnorm(cam(:, 2:3) - fbg(:, 2:3), 2, 2);
    errors.out_plane = vecnorm(cam(:,[1, 3]) - fbg(:, [1,3]), 2, 2);
    
    errors.RMSE = sqrt(mean(errors.L2.^2));
    
end


function plotShapesInImage(left_img, right_img, cam_pts, fbg_pts, stereoParams)
    px_offset = [80 5];

    % project the points into the image frames
    cam_pts_l = worldToImage(stereoParams.CameraParameters1, eye(3), zeros(3,1), ...
                cam_pts, 'ApplyDistortion', true) + px_offset;
    fbg_pts_l = worldToImage(stereoParams.CameraParameters1, eye(3), zeros(3,1), ...
                fbg_pts, 'ApplyDistortion', true) + px_offset;
            
    cam_pts_r = worldToImage(stereoParams.CameraParameters2, stereoParams.RotationOfCamera2,...
                stereoParams.TranslationOfCamera2, cam_pts, 'ApplyDistortion', true) + px_offset;
    fbg_pts_r = worldToImage(stereoParams.CameraParameters2, stereoParams.RotationOfCamera2,...
                stereoParams.TranslationOfCamera2, fbg_pts, 'ApplyDistortion', true) + px_offset;
            
    lr_img = [left_img, right_img];
    
    imshow(lr_img); hold on;
    plot(cam_pts_l(:,1), cam_pts_l(:,2), 'g-', 'LineWidth', 2); hold on;
    plot(fbg_pts_l(:,1), fbg_pts_l(:,2), 'r-', 'linewidth', 2); hold on;
    
    plt_cam_pts_r = cam_pts_r + [size(left_img, 2), 0];
    plt_fbg_pts_r = fbg_pts_r + [size(left_img, 2), 0];
    plot(plt_cam_pts_r(:,1), plt_cam_pts_r(:,2), 'g-', 'linewidth', 2); hold on;
    plot(plt_fbg_pts_r(:,1), plt_fbg_pts_r(:,2), 'r-', 'linewidth', 2); hold off;
    legend({'stereo', 'FBG'}, 'location', 'south');
    
end
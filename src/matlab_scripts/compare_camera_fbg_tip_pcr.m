%% compare_camera_fbg_tip_pcr.m
% 
% this is a script to compare shape sensing methods to FBG shape sensing
% using the tip-aligned point-cloud registration
%
% - written by: Dimitri Lezcano
clear; clc
%% Set-Up
% directories to iterate throughn ( the inidividual trials )
expmt_dir = "../../data/3CH-4AA-0004/2021-09-29_Insertion-Expmt-1/";
trial_dirs = dir(fullfile(expmt_dir, "Insertion*/"));
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories and "0" directory
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% stereo parameters
stereoparam_dir = "../../amiro-cv/calibration/Stereo_Camera_Calibration_02-08-2021/6x7_5mm/";
stereoparam_file = stereoparam_dir + "calibrationSession_params-error.mat";
stereoParams = load(stereoparam_file).stereoParams;

% FBG reliability weight options
use_weights = true;

% saving options
save_bool = false;
fileout_base = "Jig-Camera-Comp";
if use_weights == true
    fileout_base = fileout_base + "_FBG-weights";
end

% directory separation
if ispc
    dir_sep = '\';
else
    dir_sep = '/';
end

% 3D point file names
camera_pos_file = "left-right_3d-pts.csv";
if use_weights == true
    fbg_pos_file = "FBGdata_FBG-weights_3d-position.xls";
else
    fbg_pos_file = "FBGdata_3d-position.xls";
end

% arclength options
ds = 0.5;

%% Create the base table
% setup the columns
col_names = {'Ins_Hole', 'L', 'cam_shape', 'fbg_shape', 'L_cam', 'L_fbg',...
             'Pose_nc', 'RMSE', 'MaxError', 'MeanInPlane', 'MaxInPlane', ...
             'MeanOutPlane', 'MaxOutPlane'};
Ncols_err = numel(col_names) - find(strcmp(col_names, "RMSE")) + 1;
col_types = cat(2, {'uint8', 'double', 'double', 'double', 'double', 'double', 'double'},...
                repmat({'double'}, 1, Ncols_err));
col_units = cat(2, {'', 'mm', 'mm', 'mm', 'mm', 'mm' ,'mm'},...
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
                           {hole_num, L, cam_pos_interp, fbg_pos_interp,...
                            arclen_cam, arclen_fbg, F_nc, ...
                            errors.RMSE, max(errors.L2), ...
                            mean(errors.in_plane), max(errors.in_plane),...
                            mean(errors.out_plane), max(errors.out_plane)}];
    
    
end
fbg_cam_compare_tbl = sortrows(fbg_cam_compare_tbl, [1,2]);

%% Process each trial
return;
dir_prev = "";
arclength_tbl = array2table(zeros(length(trial_dirs), 4), 'VariableNames', ...
    {'hole_num', 'actual', 'FBG', 'camera'});
for i = 1:length(trial_dirs)
    tic; 
    % trial operations
    L = str2double(trial_dirs(i).name);
    re_ret = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
    hole_num = str2double(re_ret{1}{1});
    
    % trial directory
    d = strcat(trial_dirs(i).folder,dir_sep, trial_dirs(i).name, dir_sep);
    fbg_file = d + fbg_pos_file;
    camera_file = d + camera_pos_file;
    left_file = d + "left.png";
    right_file = d + "right.png";
    
    % load in the matrices
    fbg_pos = readmatrix(fbg_file)';
    camera_pos = readmatrix(camera_file);
    camera_pos = camera_pos(:,1:3);
    
    % get the arclengths of each curve
    arclen_fbg = arclength(fbg_pos);
    arclen_camera = arclength(camera_pos);
    
    arclength_tbl{i,:} = [hole_num, L, arclen_fbg, arclen_camera];
    fprintf("Arclengths (actual, FBG, Camera) [mm]: %.2f, %.2f, %.2f\n", L, arclen_fbg, arclen_camera);
    
    % perform point cloud registration from camera to needle coordinates
    [R_nc, p_nc, ...
     camera_pos_interp, s_cam,...
     fbg_pos_interp, s_fbg,...
     camera_pos_interp_tf, fbg_pos_interp_tf] = pcr_Cam2FBG(camera_pos, fbg_pos, ds, -1);
    F_nc = makeSE3(R_nc, p_nc);
    
    % read in images and rectify
    left_img = imread(left_file);
    right_img = imread(right_file);
    px_offset = [80 5]; % pixel offset for extrinsics (eye-balled)
    left_rect = undistortImage(left_img, stereoParams.CameraParameters1);
    right_rect = undistortImage(right_img, stereoParams.CameraParameters2);
    
    % project the points into the left and right iamges
%     cam_pts_l = worldToImage(stereoParams.CameraParameters1, eye(3), zeros(3,1), ...
%                 camera_pos, 'ApplyDistortion', false) + px_offset;
%     fbg_pts_l = worldToImage(stereoParams.CameraParameters1, eye(3), zeros(3,1), ...
%                 fbg_pos_interp_tf, 'ApplyDistortion', false) + px_offset;
%             
%     cam_pts_r = worldToImage(stereoParams.CameraParameters2, stereoParams.RotationOfCamera2,...
%                 stereoParams.TranslationOfCamera2, camera_pos, 'ApplyDistortion', false) + px_offset;
%     fbg_pts_r = worldToImage(stereoParams.CameraParameters2, stereoParams.RotationOfCamera2,...
%                 stereoParams.TranslationOfCamera2, fbg_pos_interp_tf, 'ApplyDistortion', false) + px_offset;
    
    cam_pts_l = worldToImage(stereoParams.CameraParameters1, eye(3), zeros(3,1), ...
                camera_pos, 'ApplyDistortion', true) + px_offset;
    fbg_pts_l = worldToImage(stereoParams.CameraParameters1, eye(3), zeros(3,1), ...
                fbg_pos_interp_tf, 'ApplyDistortion', true) + px_offset;
            
    cam_pts_r = worldToImage(stereoParams.CameraParameters2, stereoParams.RotationOfCamera2,...
                stereoParams.TranslationOfCamera2, camera_pos, 'ApplyDistortion', true) + px_offset;
    fbg_pts_r = worldToImage(stereoParams.CameraParameters2, stereoParams.RotationOfCamera2,...
                stereoParams.TranslationOfCamera2, fbg_pos_interp_tf, 'ApplyDistortion', true) + px_offset;
              
    % error analysis
    N_overlap = min(size(camera_pos_interp_tf,1), size(fbg_pos_interp,1));
    errors = error_analysis(camera_pos_interp_tf(end-N_overlap+1:end,:),...
                            fbg_pos_interp(end-N_overlap+1:end,:));
    if numel(s_fbg) >= numel(s_cam)
        s_max = s_fbg;
    else
        s_max = s_cam;
    end
    
    % Plotting
    %- 3-D shape 
    fig_shape_3d = figure(1);
    set(fig_shape_3d,'units','normalized','position', [0, 0.5, 1/3, .45])
    plot3(fbg_pos_interp(:,3), fbg_pos_interp(:,1), fbg_pos_interp(:,2), 'r-', 'LineWidth', 2); hold on;
    plot3(camera_pos_interp_tf(:,3), camera_pos_interp_tf(:,1), camera_pos_interp_tf(:,2), 'g-',...
        'LineWidth', 2); 
    hold off;
    legend('FBG', 'Stereo Recons.'); 
    title(sprintf('3-D shapes: Hole Num=%d, Ins. Depth=%.1f mm', hole_num, L));
    xlabel('z','fontweight', 'bold'); ylabel('x','fontweight', 'bold'); 
    zlabel('y','fontweight', 'bold');
    axis equal; grid on;
    
    
    %- 2-D shape
    fig_shape_2d = figure(2);
    set(fig_shape_2d,'units','normalized','position', [1/3, 0.5, 1/3, .45] )
    %-- in-plane
    subplot(2,1,1);
    plot(fbg_pos_interp(:,3), fbg_pos_interp(:,2), 'r-', 'LineWidth', 2); hold on;
    plot(camera_pos_interp_tf(:,3), camera_pos_interp_tf(:,2), 'g-', 'LineWidth', 2);
    hold off;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    %-- out-of-plane
    subplot(2,1,2);
    plot(fbg_pos_interp(:,3), fbg_pos_interp(:,1), 'r-', 'LineWidth', 2); hold on;
    plot(camera_pos_interp_tf(:,3), camera_pos_interp_tf(:,1), 'g-', 'LineWidth', 2);
    hold off;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    legend('FBG', 'Stereo Recons.'); 
    
    sgtitle(sprintf('2-D shapes: Hole Num=%d, Ins. Depth=%.1f mm', hole_num, L));
    
    %- error plots
    fig_err = figure(3);
    s_sub = s_max(end-N_overlap+1:end);
    set(fig_err,'units','normalized','position', [2/3, 0.5, 1/3, .45])
    plot(s_max, 0.5 * ones(size(s_max)), 'r--', 'DisplayName', '0.5 mm'); hold on;
    plot(s_sub, errors.L2, 'DisplayName', 'L2 Distance'); 
    plot(s_sub, errors.dx, 'DisplayName', 'x-component');
    plot(s_sub, errors.dy, 'DisplayName', 'y-component');
    plot(s_sub, errors.dz, 'DisplayName', 'z-component');
    plot(s_sub, errors.in_plane, 'DisplayName', 'in-plane');
    plot(s_sub, errors.out_plane, 'DisplayName', 'out-of-plane');
    hold off;
    xlabel('s [mm]', 'fontweight', 'bold');
    ylabel('error [mm]','fontweight', 'bold');
    xlim([0, 1.1*max(s_max)]); ylim([0, max([1.1 * errors.L2', 1])]);
    legend(); grid on;
    title(sprintf('Errors: Hole Num=%d, Ins. Depth=%.1f mm', hole_num, L));
    
    % cumulative stereo insertion plots
    % - 3D total
    fig_cum_3d = figure(4);
    set(fig_cum_3d,'units','normalized','position', [0, 0.0, 1/3, .45])
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot3(camera_pos_interp(:,1), camera_pos_interp(:,2), camera_pos_interp(:,3),...
        'linewidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('x [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold'); 
    zlabel('z [mm]', 'FontWeight', 'bold');
    legend('Location', 'southeastoutside');  
    axis equal; grid on;
    view([15, 30])
    title(sprintf("Insertion #%d | Stereo Reconstruction", hole_num));
    
    % - 2D total
    fig_cum_2d = figure(5);
    set(fig_cum_2d,'units','normalized','position', [1/3, 0.0, 1/3, .45] )
    subplot(3,1,1);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(camera_pos_interp(:,2), camera_pos_interp(:,3), ...
        'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('y [mm]', 'FontWeight', 'bold'); ylabel('z [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    zl = ylim;
    
    subplot(3,1,2);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(camera_pos_interp(:,1), camera_pos_interp(:,3), 'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('x [mm]', 'FontWeight', 'bold'); ylabel('z [mm]', 'FontWeight', 'bold');
    grid on; ylim(zl);
    
    subplot(3,1,3);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(camera_pos_interp(:,1), camera_pos_interp(:,2), 'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('x [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    legend()
    sgtitle(sprintf("Insertion #%d | Stereo Reconstruction", hole_num));
    
    fig_project = figure(6);
    set(fig_project, 'units','normalized','position', [1/3, 0.3, 2/3, .55] );
    imshow([left_img, right_img]); hold on;
    plot(cam_pts_l(:,1), cam_pts_r(:,2), 'g', 'LineWidth', 2, 'DisplayName', 'left-camera');
    plot(fbg_pts_l(:,1), fbg_pts_l(:,2), 'r', 'LineWidth', 2, 'DisplayName', 'left-fbg');
    
    plt_cam_pts_r = cam_pts_r + [size(left_img, 2), 0];
    plt_fbg_pts_r = fbg_pts_r + [size(left_img, 2), 0];
    plot(plt_cam_pts_r(:,1), plt_cam_pts_r(:,2), 'g-.', 'LineWidth', 2, 'DisplayName', 'right-camera');
    plot(plt_fbg_pts_r(:,1), plt_fbg_pts_r(:,2), 'r-.', 'LineWidth', 2, 'DisplayName', 'right-fbg');
    hold off;
    title("Left-Right Needle Shape Image Projections"); legend('Location', 'south');
    
    % time update
    t = toc;
    
    % saving
    if save_bool
        % write arclengths from each position
        T = table(L, arclen_fbg, arclen_camera, 'VariableNames', {'L', 'FBG', 'Camera'});
        writetable(T, d + fileout_base + "_arclengths-mm.txt");
        fprintf("Wrote arclengths to: '%s'\n", d + fileout_base + "_arclengths-mm.txt");
       
        % write the figures
        %- 3-D plot
        verbose_savefig(fig_shape_3d, d + fileout_base + "_3d-positions.fig");
        verbose_saveas(fig_shape_3d, d + fileout_base + "_3d-positions.png");
        
        %- 2-D plot
        verbose_savefig(fig_shape_2d, d + fileout_base + "_2d-positions.fig");
        verbose_saveas(fig_shape_2d, d + fileout_base + "_2d-positions.png");
        
        %- error plot
        verbose_savefig(fig_err, d + fileout_base + "_3d-positions-errors.fig");
        verbose_saveas(fig_err, d + fileout_base + "_3d-positions-errors.png");
        
        %- cumulative 3D
        verbose_savefig(fig_cum_3d, strcat(trial_dirs(i).folder, dir_sep, "Camera_3d-stereo-positions-cumulative.fig"))
        verbose_saveas(fig_cum_3d, strcat(trial_dirs(i).folder, dir_sep, "Camera_3d-stereo-positions-cumulative.png"))
        
        %- cumulative 2D
        verbose_savefig(fig_cum_2d, strcat(trial_dirs(i).folder, dir_sep, "Camera_2d-stereo-positions-cumulative.fig"));
        verbose_saveas(fig_cum_2d, strcat(trial_dirs(i).folder, dir_sep, "Camera_2d-stereo-positions-cumulative.png"))
        
        %- projection plots
        verbose_savefig(fig_project, d + fileout_base + "_img-projections.fig");
        verbose_saveas(fig_project, d + fileout_base + "_img-projections.png");
        
    end
    
    % update previous directory
    dir_prev = trial_dirs(i).folder;
    
    % update user
    fprintf("Finished trial: '%s' in %.2f secs.\n", d, t);
    disp(" ");
    
end

%% Arclength Error analysis
arclength_tbl.error = arclength_tbl.actual - arclength_tbl.camera;
arclength_tbl.abs_error = abs(arclength_tbl.error);

% min, max error
min_err = min(arclength_tbl.error);
max_err = max(arclength_tbl.error);
 
% mean (abs) error
mean_err = mean(arclength_tbl.error);
mean_abs_err = mean(abs(arclength_tbl.error));

% group statistics
%- by insertion distance
arclength_depth_tbl = grpstats(arclength_tbl, 'actual', ["min", "mean", "std", "max"]);

%- by insertion hole
arclength_hole_tbl = grpstats(arclength_tbl, 'hole_num', ["min", "mean", "std", "max"]);

disp("Arclength Errors ( actual - camera ) (mm)");
fprintf("Min:        %f\nMax:        %f\nMean:       %f\nAbs. Mean:  %f\n",...
        min_err, max_err, mean_err, mean_abs_err);
tbl_file_out = expmt_dir + "measured-arclength_fbg-camera.xlsx";
writetable(arclength_tbl, tbl_file_out, 'Sheet', 1)
writetable(arclength_depth_tbl, tbl_file_out, 'Sheet', 2, 'WriteRowNames', true);
writetable(arclength_hole_tbl, tbl_file_out, 'Sheet', 3, 'WriteRowNames', true);

disp(" ");

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

% saving wrappers
function verbose_savefig(fig, file)
    savefig(fig, file);
    fprintf('Saved figure: %s\n', file);
    
end

function verbose_saveas(fig, file)
    saveas(fig, file);
    fprintf('Saved image: %s\n', file);
    
end

%% compare_camera_fbg_aruco.m
% 
% this is a script to compare shape sensing methods to FBG shape sensing
%  using the ARUCO markers as a frame
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
stereoparam_file = fullfile(stereoparam_dir,"calibrationSession_params-error.mat");
stereoParams = load(stereoparam_file).stereoParams;

% ARUCO parameters (pose of ARUCO in needle frame)
aruco_dir = "../../data/aruco/";
aruco_pose_file = fullfile(aruco_dir, 'pose_needle_aruco_calibrated_manual.mat');
aruco_needle_pose = load(aruco_pose_file).pose_FA_cal;
F_update = translationToSE3([-20; -70; 0]);
% F_update = rotationToSE3(roty(pi/10)) * F_update;
% F_update = translationToSE3([-18; 0; 0]) * F_update;
% F_update = rotationToSE3(rotx(-pi/30)) * F_update;
aruco_needle_pose = F_update*aruco_needle_pose;

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


%% Process each trial
dir_prev = "";
arclength_tbl = array2table(zeros(length(trial_dirs), 4), 'VariableNames', ...
    {'hole_num', 'actual', 'FBG', 'camera'});
for i = 1:length(trial_dirs)
    tic; 
    % trial operations
    L        = str2double(trial_dirs(i).name);
    re_ret   = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
    hole_num = str2double(re_ret{1}{1});
    
    if hole_num == 1
        continue;
    end
    
    % trial directory
    d           = fullfile(trial_dirs(i).folder, trial_dirs(i).name);
    fbg_file    = fullfile(d, fbg_pos_file);
    camera_file = fullfile(d, camera_pos_file);
    left_file   = fullfile(d, "left.png");
    right_file  = fullfile(d, "right.png");
    aruco_file  = fullfile(d, "left-right_aruco-poses");
    
    % load in the matrices
    fbg_pos    = readmatrix(fbg_file)';
    camera_pos = readmatrix(camera_file);
    camera_pos = camera_pos(:,1:3);
    
    % load the aruco pose measured
    [aruco_pose_l, ~, id] = read_aruco_pose(aruco_file);
    
    % get the arclengths of each curve
    arclen_fbg    = arclength(fbg_pos);
    arclen_camera = arclength(camera_pos);
    
    arclength_tbl{i,:} = [hole_num, L, arclen_fbg, arclen_camera];
    fprintf("Arclengths (actual, FBG, Camera) [mm]: %.2f, %.2f, %.2f\n", L, arclen_fbg, arclen_camera);
    
    % interpolate and align the camera positions
    [camera_pos_interp_tf, fbg_pos_interp, pose_align, thetaz] = align_camera_needle_shapes(...
        camera_pos, fbg_pos, aruco_pose_l, aruco_needle_pose, ...
        'interpolate', true, 'sCutoff', [30, inf], 'ds', 0.5, 'RotateZ', false);
    
    % Invert the pose to go back to camera coordiantes
    camera_pos_interp = [camera_pos_interp_tf, ones(size(camera_pos_interp_tf,1),1)] * finv(pose_align)';
    camera_pos_interp = camera_pos_interp(:,1:3);
%     camera_pos_interp = camera_pos;
    fbg_pos_interp_tf = [fbg_pos_interp, ones(size(fbg_pos_interp,1),1)] * finv(pose_align)';
    fbg_pos_interp_tf = fbg_pos_interp_tf(:,1:3);
    
    [~,~,s_fbg] = arclength(fbg_pos_interp);
    [~,~,s_cam] = arclength(camera_pos_interp_tf);
    
    if length(s_fbg) >= length(s_cam)
        s_max = s_fbg;
    else%  length(s_fbg) < length(s_camera)
        s_max = s_cam;
    end
    
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
        writetable(T, fullfile(d, fileout_base + "_arclengths-mm.txt"))
        fprintf("Wrote arclengths to: '%s'\n", fullfile(d, fileout_base + "_arclengths-mm.txt"));
       
        % write the figures
        %- 3-D plot
%         verbose_savefig(fig_shape_3d, d + fileout_base + "_3d-positions.fig");
%         verbose_saveas(fig_shape_3d, d + fileout_base + "_3d-positions.png");
        savefigas(fig_shape_3d, fullfile(d, strcat(fileout_base,"_3d-positions")), 'Verbose', true);
        
        %- 2-D plot
%         verbose_savefig(fig_shape_2d, d + fileout_base + "_2d-positions.fig");
%         verbose_saveas(fig_shape_2d, d + fileout_base + "_2d-positions.png");
        savefigas(fig_shape_2d, fullfile(d, strcat(fileout_base,"_2d-positions-errors")), 'Verbose', true);
        
        %- error plot
%         verbose_savefig(fig_err, d + fileout_base + "_3d-positions-errors.fig");
%         verbose_saveas(fig_err, d + fileout_base + "_3d-positions-errors.png");
        savefigas(fig_err, fullfile(d, strcat(fileout_base,"_3d-positions-errors")), 'Verbose', true);
        
        %- cumulative 3D
%         verbose_savefig(fig_cum_3d, strcat(trial_dirs(i).folder, dir_sep, "Camera_3d-stereo-positions-cumulative.fig"))
%         verbose_saveas(fig_cum_3d, strcat(trial_dirs(i).folder, dir_sep, "Camera_3d-stereo-positions-cumulative.png"))
        savefigas(fig_cum_3d, fullfile(trial_dirs(i).folder, "Camera_3d-stereo-positions-cumulative"));
        
        %- cumulative 2D
%         verbose_savefig(fig_cum_2d, strcat(trial_dirs(i).folder, dir_sep, "Camera_2d-stereo-positions-cumulative.fig"));
%         verbose_saveas(fig_cum_2d, strcat(trial_dirs(i).folder, dir_sep, "Camera_2d-stereo-positions-cumulative.png"))
        savefigas(fig_cum_2d, fullfile(trial_dirs(i).folder, "Camera_2d-stereo-positions-cumulative"));
        
        %- projection plots
%         verbose_savefig(fig_project, d + fileout_base + "_img-projections.fig");
%         verbose_saveas(fig_project, d + fileout_base + "_img-projections.png");
        savefigas(fig_cum_3d, fullfile(trial_dirs(i).folder, "Camera_3d-stereo-positions-cumulative"));
        
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
% close all;


%% Helper functions
% error analysis
function errors = error_analysis(nurbs, jig)
% measures error metrics from each points
    
    % L2 distance
    errors.L2 = vecnorm(nurbs - jig, 2, 2); 
    
    % component-wise error
    errors.dx = abs(nurbs(:,1) - jig(:,1));
    errors.dy = abs(nurbs(:,2) - jig(:,2));
    errors.dz = abs(nurbs(:,3) - jig(:,3));
    
    % in/out-plane error (assume in-plane is yz and out-plane is xz)
    errors.in_plane = vecnorm(nurbs(:, 2:3) - jig(:, 2:3), 2, 2);
    errors.out_plane = vecnorm(nurbs(:,[1, 3]) - jig(:, [1,3]), 2, 2);
    
end

    
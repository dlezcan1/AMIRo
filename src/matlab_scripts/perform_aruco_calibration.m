%% perform_aruco_calibration.m
%
% script to perform the calibration for ARUCO alignment
%
% - written by: Dimitri Lezcano

%% Parameters to Change
% experimental data
expmt_dir = "../../amiro-cv/aruco/aruco_needle_calibration"; 

% saving
save_bool = true;

% error threshold to exclude (max distance)
error_thresh = 0.18;

%% Set-up
% trial regexp
trial_pattern = "Insertion([0-9])[/,\\]([0-9]+)";

% Insertion Experiment
trial_dirs = dir(fullfile(expmt_dir, "Insertion*/*"));
mask_remove = strcmp({trial_dirs.name}, '0') | strcmp({trial_dirs.name}, '.') | strcmp({trial_dirs.name}, '..');
mask_remove = mask_remove | ~[trial_dirs.isdir];
trial_dirs = trial_dirs(~mask_remove);

% experimental files
camera_shape_file = "left-right_3d-pts.csv";
aruco_pose_file = "left-right_aruco-poses.csv";

%% Iterate through the trials to compile results
pose_FA_mat = zeros(4,4,numel(trial_dirs));
pose_CA_mat = zeros(4,4,numel(trial_dirs));
pose_FC_mat = zeros(4,4,numel(trial_dirs));
figure(4);
for i = 1:numel(trial_dirs)
    % prepare files
    d = fullfile(trial_dirs(i).folder, trial_dirs(i).name);
    camshape_file_i = fullfile(d, camera_shape_file);
    arucopose_file_i = fullfile(d, aruco_pose_file);
    
    % grab the insertion results
    re_res = regexp(d, trial_pattern, 'tokens');
    ins_hole = str2double(re_res{1}{1}); 
    L = str2double(re_res{1}{2});
    
    % determine straight needle pose
    [p_needle, R_needle] = straight_needle(L);
    
    % load stereo points and aruco pose
    p_stereo = readmatrix(camshape_file_i);
    [pose_aruco_l, pose_aruco_r, ~] = read_aruco_pose(arucopose_file_i);
    if any(isnan(pose_aruco_l)) % only use left aruco (for now at least)
        
        continue;
    else
        pose_CA_i = pose_aruco_l;
    end
    
    % point cloud registration tip between stereo and straight needle
    [pose_FC_i, errors_i, p_stereo_interp, p_needle_interp] = point_cloud_reg_tip_interp(p_stereo, p_needle);
    if max(errors_i.norm) >= error_thresh
        continue;
    end
    fprintf("%d: max error = %.4f\n", i, max(errors_i.norm));
    p_stereo_tf = [p_stereo, ones(size(p_stereo,1),1)] * pose_FC_i';
%     figure(4);
%     plot3(p_stereo_tf(:,3), p_stereo_tf(:,1), p_stereo_tf(:,2),'LineWidth', 3); hold on;
%     plot3(p_needle(:,3), p_needle(:,1), p_needle(:,2),'LineWidth', 3); hold on;
%     axis equal; grid on; hold off;
%     xlabel('z'); ylabel('x'); zlabel('y');
%     legend('stereo', 'needle');
%     pause(2);
    
    % compute pose_AF for this iteration
    pose_AF_i =  pose_FC_i * pose_CA_i;
    pose_FA_mat(:,:,i) = pose_AF_i;
    pose_FC_mat(:,:,i) = pose_FC_i;
    pose_CA_mat(:,:,i) = pose_CA_i;
    
    
end

% remove the skipped trials
skipped_trials = all(pose_FA_mat == 0, [1,2]);
pose_FA_mat(:,:,skipped_trials) = [];
pose_FC_mat(:,:,skipped_trials) = [];
pose_CA_mat(:,:,skipped_trials) = [];

%% Plotting
figure(1);
for i = 1:size(pose_FA_mat,3)
    if all(pose_FA_mat(:,:,i) == 0)
        continue;
    end
    figure(1);
    plotf(pose_FA_mat(:,:,i), 'scale', 12); hold on;
    title("T_A^F");
    
    figure(2);
    plotf(pose_FC_mat(:,:,i), 'scale', 12); hold on;
    title("T_C^F");
    
    figure(3);
    plotf(pose_CA_mat(:,:,i), 'scale', 12); hold on;
    title("T_A^C");
end

figure(1); hold off;
figure(2); hold off;
figure(3); hold off;

%% Perform the calibration
% remove outliers
t_FA_mat = squeeze(pose_FA_mat(1:3,4,:));
outliers = any(isoutlier(t_FA_mat,2),1);
fprintf("# outliers: %d\n", sum(outliers));
pose_FA_mat_clean = pose_FA_mat(:,:,~outliers); % remove outliers
pose_FC_mat_clean = pose_FC_mat(:,:,~outliers);
pose_CA_mat_clean = pose_CA_mat(:,:,~outliers);

% plot the outlier-less poses
for i = 1:size(pose_FA_mat_clean,3)
    if all(pose_FA_mat(:,:,i) == 0)
        continue;
    end
    figure(4);
    plotf(pose_FA_mat_clean(:,:,i), 'scale', 12); hold on;
    title("T_A^F no outliers");
    
    figure(5);
    plotf(pose_FC_mat(:,:,i), 'scale', 12); hold on;
    title("T_C^F no outliers");
    
    figure(6);
    plotf(pose_CA_mat(:,:,i), 'scale', 12); hold on;
    title("T_A^C no outliers")
    
end
figure(4); hold off;
figure(5); hold off;
figure(6); hold off;

% Take the mean translation and rotation
R_FA_mat_clean = pose_FA_mat_clean(1:3,1:3,:);
t_FA_mat_clean = squeeze(pose_FA_mat_clean(1:3,4,:));

R_FA_cal = rotation_geodesic_mean(R_FA_mat_clean, 'MaxLoops', 10000, 'Tolerance', 1e-8);
t_FA_cal = mean(t_FA_mat_clean,2);
r_FA_cal = rotationMatrixToVector(R_FA_cal);
pose_FA_cal_vect = [r_FA_cal, t_FA_cal'];
pose_FA_cal = [R_FA_cal, t_FA_cal; zeros(1,3),1];
figure(1); hold on; plotf(pose_FA_cal, 'scale', 12); hold off;
figure(4); hold on; plotf(pose_FA_cal, 'scale', 12); hold off;

%% Saving
if save_bool
    outfile_base = fullfile(expmt_dir, 'pose_needle_aruco_calibrated');
    
    % write matrix
    save(strcat(outfile_base, '.mat'), 'pose_FA_cal');
    fprintf("Saved pose to: %s\n", strcat(outfile_base, '.mat'));
    
    % write 4x4 pose to file
    writematrix(pose_FA_cal, strcat(outfile_base,'_4x4.csv'), ...
        'Delimiter', ',');
    fprintf("Saved pose to: %s\n", strcat(outfile_base,'_4x4.csv'));
    
    % write vectorized pose to file
    writematrix(pose_FA_cal_vect, strcat(outfile_base, '_vect.csv'),...
        'Delimiter', ',');
    fprintf("Saved pose to: %s\n", strcat(outfile_base,'_vect.csv'));
    
end
    
    


%% Helper Functions
function [p, R] = straight_needle(L, ds)
    arguments
        L double {mustBePositive};
        ds double = 0.5;
    end
    
    z = 0:ds:L;
    p = [zeros(length(z),2), z'];
    
    R = permute(repmat(eye(3),1,1,size(p,2)), [3,1,2]);
    
end

function [pose, errors, shape_cam_interp, shape_needle_interp] = point_cloud_reg_tip_interp(shape_cam, shape_needle, ds)
    arguments
        shape_cam (:,3);
        shape_needle (:,3);
        ds double = 0.5;
    end
    
    arclen_c = arclength(shape_cam);
    arclen_n = arclength(shape_needle);
    
    s_c = flip(arclen_c:-ds:0);
    s_n = 0:ds:arclen_n;
    
    N = min(numel(s_c), numel(s_n));
    
    shape_cam_interp = interp_pts(shape_cam, s_c);
    shape_needle_interp = interp_pts(shape_needle, s_n);
    
    % register camera -> Needle
    [R, p] = point_cloud_reg_tip(shape_cam_interp(end-N+1:end,:),...
                                 shape_needle_interp(end-N+1:end,:));
                             
    pose = [R, p; zeros(1,3), 1];
    
    shape_cam_interp_tf = shape_cam_interp(end-N+1:end,:) * R' + p';
    
    errors.diff = shape_cam_interp_tf - shape_needle_interp(end-N+1:end,:);
    errors.norm = vecnorm(errors.diff, 2,2);
    
    
end
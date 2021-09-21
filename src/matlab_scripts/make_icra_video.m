clear all; close all;
configure_env on;
%% Set-up
expmt_dir = '../../data/3CH-4AA-0004/ICRA2022_video';
stereo_video_points_dir = fullfile(expmt_dir, 'stereo_video');
trial_dirs = dir(fullfile(expmt_dir, 'Insertion*/*'));
trial_dirs = trial_dirs([trial_dirs.isdir]);
mask_remove = strcmp({trial_dirs.name}, '.') | strcmp({trial_dirs.name}, '..') | strcmp({trial_dirs.name}, '0');
trial_dirs  = trial_dirs(~mask_remove);

% video files
left_video_file = fullfile(expmt_dir, 'left_video - Copy.avi');
right_video_file = fullfile(expmt_dir, 'right_video - Copy.avi');

% stereo parameters
stereo_param_file = '../../amiro-cv/calibration/Stereo_Camera_Calibration_02-08-2021/6x7_5mm/calibrationSession_params-error.mat';
stereoParams = load(stereo_param_file).stereoParams;

% needle parameters
needleparams = load("../../shape-sensing/shapesensing_needle_properties_18G.mat");

% prediction params
L_pred = 120;
p = 0.592;
ds = 0.5;

%% Get the FBG needle shapes for the different frame numbers
expmt_table = table('Size', [0,6], 'VariableTypes', {'uint64', 'double', 'double', 'double', 'double', 'string'},...
    'VariableNames', {'frame', 'fbg_pos','fbg_pos_pred', 'L_ref', 'L_pred', 'dir'});
for i = 1:numel(trial_dirs)
    d = fullfile(trial_dirs(i).folder,trial_dirs(i).name);
    frame_i = readmatrix(fullfile(d, 'frame_num.txt'));
    
    % get the fbg shape
    fbg_pos_i = readmatrix(fullfile(d, 'FBGdata_FBG-weights_3d-position.xls'));
    
    % get fbg shape parameters
    fbg_params_i = readtable(fullfile(d, 'FBGdata_FBG-weights_3d-params.txt'));
    kc = fbg_params_i.kc;
    w_init = [fbg_params_i.w_init_1; fbg_params_i.w_init_2; fbg_params_i.w_init_3];
    theta0 = fbg_params_i.theta0;
    L_ref = fbg_params_i.L;
    
    % perform the FBG shape prediction
    [~, fbg_pos_i, ~] = fn_intgEP_1layer_Dimitri(kc, w_init, theta0, L_ref, 0, ds, needleparams.B, needleparams.Binv);
    fbg_pos_pred_i = predict_insertion_singlelayer(L_pred, L_ref, kc, w_init, needleparams, ...
                p, 'optim_lb', -1.0, 'optim_ub', 1.0, 'theta0', theta0, ...
                'optim_display', 'none');
    
    expmt_table = [expmt_table; {frame_i, fbg_pos_i, fbg_pos_pred_i, L_ref, L_pred, d}];
    
end

disp(expmt_table);


%% Load the videos
left_video = VideoReader(left_video_file);
right_video = VideoReader(right_video_file);
start_frame = 3000;

% write out video
v = VideoWriter('stereo_video_processed-3000.avi','Motion JPEG AVI');
v.Quality = 95;
v.FrameRate = left_video.FrameRate;
open(v);

fig_vid = figure(1);
if start_frame > 0
    read(left_video, start_frame); read(right_video, start_frame);
end
fbg_pos = zeros(0,3);
fbg_pred_pos = zeros(0,3);
while hasFrame(left_video) && hasFrame(right_video)
    % Read the frame
    frame_l = readFrame(left_video);
    frame_r = readFrame(right_video);
    frame_counter = round(left_video.CurrentTime * left_video.FrameRate);

    % determine if there are any points
    frame_points = fullfile(stereo_video_points_dir, ...
                            strcat(num2str(frame_counter), '.csv'));
   
    % show the images
    frame_lr = cat(2, frame_l, frame_r);
    fprintf("Frame: %5d\n", frame_counter);
    imshow(frame_lr);
    
    remove_frames = (1320 <=frame_counter) && (frame_counter <= 2060);
    remove_frames = remove_frames || (frame_counter >= 3100);
    
    % grab and plot the 3D stereo points
    if isfile(frame_points) && ~remove_frames
        stereo_points = readmatrix(frame_points);
        [left_imgpoints, right_imgpoints] = project_into_stereoplane(stereo_points, stereoParams, size(frame_r));
        fprintf("Detected Points! ");
    
        hold on; plot(left_imgpoints(:,1), left_imgpoints(:,2),'g-', 'LineWidth', 3);
        hold on; plot(right_imgpoints(:,1), right_imgpoints(:,2),'g-', 'LineWidth', 3);
    end
    
    % reset FBG shapes
    if frame_counter == 1935 % need to update the counter
        fbg_pos = zeros(0,3);
        fbg_pred_pos = zeros(0,3);
    end
    
    % see if there is an FBG data points
    if any(expmt_table.frame == frame_counter) % update the needle shape.
       fbg_pos = expmt_table.fbg_pos{expmt_table.frame == frame_counter};
       fbg_pos_pred = expmt_table.fbg_pos_pred{expmt_table.frame == frame_counter};
       
       % Determine rigid body transform from fbg_pos to camera position
       [R, t] = camera_rigid_body_tf(stereo_points', fbg_pos);
       fbg_pos_tf = R * fbg_pos + t;
       fbg_pos_pred_tf = R * fbg_pos_pred + t;
       
       % project into the image plane
       [fbg_pos_tf_imgl, fbg_pos_tf_imgr] = project_into_stereoplane(fbg_pos_tf', ...
                                                stereoParams, size(frame_r));
       [fbg_pos_pred_tf_imgl, fbg_pos_pred_tf_imgr] = project_into_stereoplane(fbg_pos_pred_tf', ...
                                                stereoParams, size(frame_r));
       
    end
    
    if size(fbg_pos,1) > 0 && size(fbg_pos_pred, 1) > 0 % we have an FBG needle shape
       hold on; plot(fbg_pos_tf_imgl(:,1), fbg_pos_tf_imgl(:,2), 'r-', 'LineWidth', 3);
       hold on; plot(fbg_pos_tf_imgr(:,1), fbg_pos_tf_imgr(:,2), 'r-', 'LineWidth', 3);
       
       hold on; plot(fbg_pos_pred_tf_imgl(:,1), fbg_pos_pred_tf_imgl(:,2), 'm-', 'LineWidth', 1.5);
       hold on; plot(fbg_pos_pred_tf_imgr(:,1), fbg_pos_pred_tf_imgr(:,2), 'm-', 'LineWidth', 1.5);
    end
        
%     title(sprintf("Frame: %d", frame_counter));
    vid_frame = getframe(fig_vid);
    writeVideo(v, vid_frame);
    pause(1/left_video.FrameRate);
    hold off;
end
close(v);

%% Helper Functions
function [R, p] = camera_rigid_body_tf(shape_cam, shape_fbg, ds)
    arguments
        shape_cam (3,:);
        shape_fbg (3,:);
        ds double = 0.5;
    end
        
    % find each curve arclength
    arclen_cam = arclength(shape_cam');
    arclen_fbg = arclength(shape_fbg');
    
    % generate the arclengths
    s_cam = flip(arclen_cam:-ds:0);
    s_fbg = 0:ds:arclen_fbg;
    
    N = min(numel(s_cam(s_cam>40)), numel(s_fbg(s_fbg > 40)));
    
    % interpolate the shapes
    shape_cam_interp = interp_pts(shape_cam', s_cam);
    shape_fbg_interp = interp_pts(shape_fbg', s_fbg);
    
    [R, p] = point_cloud_reg_tip(shape_fbg_interp(end-N+1:end,:),...
                                 shape_cam_interp(end-N+1:end,:));
    

end

function [left_imgpoints, right_imgpoints] = project_into_stereoplane(points, stereoParams, img_size)
    arguments
        points (:,3)
        stereoParams;
        img_size;
    end
    % image offset
    imgpt_offset = [78, 0];
    
    left_imgpoints = worldToImage(stereoParams.CameraParameters1, ...
                        eye(3), zeros(3,1), ...
                        points, 'ApplyDistortion', true) + imgpt_offset;
                    
    right_imgpoints = worldToImage(stereoParams.CameraParameters2, ...
                        stereoParams.RotationOfCamera2, stereoParams.TranslationOfCamera2,...
                        points, 'ApplyDistortion', true) + [img_size(2), 0] + imgpt_offset;
                    
end
    
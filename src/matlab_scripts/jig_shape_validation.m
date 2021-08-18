%% jig_shape_validation.m
%
% this is a script to perform shape validation using the jig from an fbg needle
%
% - written by: Dimitri Lezcano

set(0,'DefaultAxesFontSize',24);
clear;
%% Set-up 
% options
save_bool = true;
process_only = false;

% python set-up
pydir = fullfile('../');

if count(py.sys.path, pydir) == 0
    insert(py.sys.path, int32(0), pydir);
end

% file set-up
directory = "../../data/3CH-4AA-0004/";
fbgneedle_param = fullfile(directory, ...
    "needle_params_08-16-2021_Jig-Calibration_clinically-relevant_weights.json"); 

datadir = fullfile(directory, "08-16-2021_Jig-Calibration/"); % calibration-validation data
data_mats_file = "Jig-Calibration-Validation-Data.xlsx"; % all data
proc_data_sheet = 'Calibration Validation Dataset';

% paramteter set-up
jig_offset = 26.0; % the jig offset of full insertion
% AA_weights = [ 0.613865, 0.386135, 0.000000, 0.000000 ]; [ 0.774319, 0.090095, 0.135586 ]; % [AA1, AA2, AA3, AA4] reliability weighting
fig_save_file = "Jig_Shape_fit";
if contains(fbgneedle_param, 'clinically-relevant')
    data_mats_file = strcat('clinically-relevant_', data_mats_file);
    fig_save_file = strcat('clinically-relevant_', fig_save_file);
elseif contains(fbgneedle_param, '_all')
    data_mats_file = strcat('all_', data_mats_file);
    fig_save_file = strcat('all_',fig_save_file);
else
    data_mats_file = strcat('jig_', data_mats_file);
    fig_save_file = strcat('jig_',fig_save_file);
end

if contains(fbgneedle_param, 'weighted')
    data_mats_file = strrep(data_mats_file, '.xlsx', '_weighted.xlsx');
    fig_save_file = strcat(fig_save_file, '_weighted');
end

% add the data directories
data_mats_file = fullfile(datadir, data_mats_file);
fig_save_file = fullfile(datadir, fig_save_file);

%% Load FBGNeedle python class
fbg_needle = py.sensorized_needles.FBGNeedle.load_json(fbgneedle_param);
disp("FBGNeedle class loaded.")

% channel list
ch_list = 1:double(fbg_needle.num_channels);
CH_list = "CH" + ch_list;
aa_list = 1:double(fbg_needle.num_activeAreas);
AA_list = "AA" + aa_list; % the "AAX" string version
ret     = fbg_needle.generate_chaa();
CH_AA   = cellfun(@char,cell(ret{1}),'UniformOutput',false);

% check for AA_weights
if py.len(fbg_needle.weights) > 0
    AA_weights = [];
    for AA_i = AA_list
        AA_weights = [AA_weights, fbg_needle.weights{fbg_needle.aa_loc(AA_i)}];
    end
    
    fig_save_file = fig_save_file + "_weights";
else
    AA_weights = [];
end

%% load the data matrices TODO
data_mats = struct();
tbl = readtable(data_mats_file, 'Sheet', proc_data_sheet, ...
        'VariableNamingRule', 'preserve', 'ReadRowNames', true); % remove the first column (exp #)
for AA_i = AA_list
    CH_AA_i = CH_AA(contains(CH_AA, AA_i));
    curv_head = {'type', 'Curvature (1/m)', 'Curvature_x (1/m)', 'Curvature_y (1/m)', ...
        [char(AA_i), ' Predicted Curvature_x (1/m)'], [char(AA_i), ' Predicted Curvature_y (1/m)']};
    head_AA_i = cat(2, curv_head, CH_AA_i);
    data_mats.(AA_i) = tbl(:,head_AA_i);
    disp(AA_i + " loaded.");
end
disp(' ');

%% determine the calculated curvature for each AA
for AA_i = AA_list
    CH_AA_i = CH_AA(contains(CH_AA, AA_i));
    cal_mat = double(fbg_needle.aa_cal(AA_i).T);
    signals = data_mats.(AA_i){:,CH_AA_i};
    pred_curv = signals * cal_mat;
    data_mats.(AA_i).PredCurvX = pred_curv(:,1);
    data_mats.(AA_i).PredCurvY = pred_curv(:,2);
    data_mats.(AA_i).PredCurv = vecnorm([data_mats.(AA_i).PredCurvX, data_mats.(AA_i).PredCurvY], 2, 2);
    disp("Calculated predicted curvature vector from " + AA_i + ".");
end
disp(" ");

%% Process the data shapes
s = 0:0.5:(double(fbg_needle.length) - jig_offset); % the arclength points
num_expmts = size(data_mats.AA1,1);

% iterate over all of the experiments
shape_results = cell(num_expmts, 1);
for exp_num = 1:num_expmts
    EXPMT_i.curvature = data_mats.AA1{exp_num, 'Curvature (1/m)'};
    EXPMT_i.curv_act = [data_mats.AA1{exp_num,["Curvature_x (1/m)", "Curvature_y (1/m)"]}'; 0]/1000; % actual curvature vector
    EXPMT_i.ref_angle = rad2deg(atan2(EXPMT_i.curv_act(2), EXPMT_i.curv_act(1)));
    EXPMT_i.type = data_mats.AA1.type{exp_num};
    
    % get the measured curvatures @ each AA
    EXPMT_i.w_AA = zeros(3, length(AA_list));
    for i = 1:length(AA_list)
        AA_i = AA_list(i);
        EXPMT_i.w_AA(1:2, i) = [data_mats.(AA_i){exp_num, ["PredCurvX", "PredCurvY"]}']/1000;
        
    end
    
    % compute the approximate curvature
    EXPMT_i.w_AA_approx = const_curv_approx(EXPMT_i.w_AA, AA_weights);
    
    % compute the shapes
    EXPMT_i.r_act = const_curv_shape(EXPMT_i.curv_act, s); % the reference shape
    EXPMT_i.r_meas = const_curv_shape(EXPMT_i.w_AA_approx, s); % the measured shape
    
    shape_results{exp_num} = EXPMT_i;
end

%% Plotting
close all;
for exp_num = 1:num_expmts
    % grab the data
    expmt_i = shape_results{exp_num};
    r_act = expmt_i.r_act;
    r_meas = expmt_i.r_meas;
    err_r = error_s_positions(r_act, r_meas);
    err_inplane = error_s_positions_inplane(r_act, r_meas, expmt_i.curv_act);
    
    % start plotting
    figure('WindowStyle', 'docked');
    
%     set(gcf,'units', 'normalized', 'position', [1/8, 0.2/8, 6/8, 7/8]);
    
    % plot the z-x deformation
    subplot(3,1,1);
    plot(r_act(3,:), r_act(1,:), 'DisplayName', 'Reference', 'LineWidth', 2); hold on;
    plot(r_meas(3,:), r_meas(1,:), 'DisplayName', 'Measured', 'LineWidth', 2); hold off;
    ylabel('x [mm]', 'fontweight', 'bold'); % xlabel('z [mm]', 'fontweight', 'bold'); 
    grid on; %axis equal;
    legend('Location', 'southwest');    
    
    % plot the z-y deformation
    subplot(3,1,2);
    plot(r_act(3,:), r_act(2,:), 'DisplayName', 'Reference', 'LineWidth', 2); hold on;
    plot(r_meas(3,:), r_meas(2,:), 'DisplayName', 'Measured', 'LineWidth', 2); hold off;
    ylabel('y [mm]', 'fontweight', 'bold'); xlabel('z [mm]', 'fontweight', 'bold'); 
    grid on; %axis equal;
    
    % plot the error
    subplot(3,1,3);
    plot(s, err_r, 'k', 'LineWidth', 2, 'DisplayName', 'Total Distance'); hold on;
    plot(s, err_inplane, 'b', 'LineWidth', 2, 'DisplayName', 'In-Plane'); hold on;
    plot([0, max(s)], [0.5, 0.5], 'r--', 'LineWidth', 1.5, 'DisplayName', '0.5 mm'); hold off;
    title('Error: L2 Distance');
    ylabel('error [mm]', 'fontweight', 'bold'); xlabel('s [mm]', 'fontweight', 'bold'); 
    grid on;
    xlim([0, 1.1*max(s)]); ylim([0, max([1.1 * err_r, 1])]);
    legend('Location', 'northwest')
    
    sgtitle("\kappa = " + sprintf("%.3f at %.1f^o about z-axis | %s", expmt_i.curvature, expmt_i.ref_angle, expmt_i.type), ...
        'fontsize', 30, 'fontweight', 'bold');
    
    % saving
    if save_bool
        saveas(gcf, fig_save_file + sprintf("_k_%.3f_ang_%.1fdeg.png", expmt_i.curvature, expmt_i.ref_angle));
        disp("Saved figure: " + fig_save_file + sprintf("_k_%.3f_ang_%.1fdeg.png", expmt_i.curvature, expmt_i.ref_angle));
    end
end

%% Termination
disp("Program Terminated.");

%% Functions 
% % jig (constant curvature) shape model
% function r = const_curv_shape(w, s)
% % function to get the constant curvature shape t
% %
% % constant curvature is 1/k * v | k = curvature, v = "torque" vector
% % 
% % Input:
% %   - w: the angular deformation vector (constant)
% %   - s: the arclength coordinates N-vector
% 
%     wv = w .* ones(3, length(s));
%     
%     r = wv2r(wv, max(s));
%     
% end

% get the constant curvature vector from AA data
function w = const_curv_approx(w_aa, weights)
% function to approximate the constant curvature vector from AA data
%
% returns the mean vector across the 3-axes
%
% Input:
%   - w_aa a 3 x #AA matrix for each AA
%
    if ~exist('weights', 'var') 
        weights = ones(1, size(w_aa, 2));
    
    elseif isempty(weights)
        weights = ones(1, size(w_aa, 2));
    
    end
    
    w = sum(w_aa.*weights, 2)/sum(weights);
    
end

% error in positions as a function of arclength (s)
function err = error_s_positions(r_1, r_2)
% function to determine the error for each arclength position between two shapes
%
% returns err(i) = || r_1(:,i) - r_2(:,i) ||

    err = vecnorm(r_1 - r_2);
    
end


% error in-plane deformaiton positions as a function of arclenth (s)
% TODO: Need to add the normal (w_act) to define the plane
function err = error_s_positions_inplane(r_1, r_2, w_act)
    % calculate the positional deviation
    dr = r_1 - r_2;
    
    % subtract out the out-of-plane deformation
    if vecnorm(w_act) > 0
        n = w_act./vecnorm(w_act);
    else
        n = zeros(3,1);
    end
    dr_inplane = dr - ((dr'*n)*n')';
    
    % calculate the error
    err = vecnorm(dr_inplane);
    
end
    
    
        
        
    

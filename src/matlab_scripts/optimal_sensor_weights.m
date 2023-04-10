%% optimal_sensor_weights.m
%
% this is a script to determine the optimal sensor weighting for the 
% FBG sensors by solving a least squares problem
%
% - written by: Dimitri Lezcano

clear;
%% Set-up 
% options
save_bool = true;

% python set-up
pydir = fullfile('../');

if count(py.sys.path, pydir) == 0
    insert(py.sys.path, int32(0), pydir);
end

% file set-up
directory = "../../data/7CH-4AA-0001-MCF-even";
fbgneedle_param = fullfile( ...
    directory, ...
    "needle_params_7CH-4AA-0001-MCF-even_2023-03-29_Jig-Calibration_clinically-relevant-2_weighted.json");   % weighted calibration
fbgneedle_param_weight = strrep(fbgneedle_param, '.json', '_weights.json'); % weighted fbg parmeters

datadir = fullfile(directory, "2023-03-29_Jig-Calibration"); % calibration-validation data
data_mats_file = "Jig-Calibration-Validation-Data.xlsx"; % all data
proc_data_sheet = 'Calibration Validation Dataset';
fig_save_file = "Jig_Shape_fit";

% paramteter set-up
jig_offset = 26.0; % the jig offset of full insertion
equal_weighting_lambda = 1; % weighting of how close to equal weighting
curv_weighting = true;
handle_mcf = true; % remove central core from signal processing
mcf_channel = 4;

if contains(fbgneedle_param, '_clinically-relevant') 
%     data_mats_file = strcat('clinically-relevant_', data_mats_file);
    fig_save_file = strcat('clinically-relevant_', fig_save_file);
elseif contains(fbgneedle_param, '_all')
%     data_mats_file = strcat('all_', data_mats_file);
    fig_save_file = strcat('all_',fig_save_file);
else
%     data_mats_file = strcat('jig_', data_mats_file);
    fig_save_file = strcat('jig_',fig_save_file);
end

if contains(fbgneedle_param, '_weighted')
    data_mats_file = strrep(data_mats_file, '.xlsx', '_weighted.xlsx');
    fig_save_file = strcat(fig_save_file, '_weighted');
end

% add the data directories
data_mats_file = fullfile(datadir, data_mats_file);
fig_save_file = fullfile(datadir, fig_save_file);
disp(data_mats_file);

%% Load FBGNeedle python class
fbg_needle = py.sensorized_needles.FBGNeedle.load_json(fbgneedle_param);
disp("FBGNeedle class loaded.")

% channel list
ch_list = 1:double(fbg_needle.num_channels);
CH_list = "CH" + ch_list;
aa_list = 1:double(fbg_needle.num_aa);
AA_list = "AA" + aa_list; % the "AAX" string version
ret     = fbg_needle.generate_chaa();
CH_AA   = cellfun(@char,cell(ret{1}),'UniformOutput',false);
if handle_mcf
    try
        disp("MCF Needle!");
        mask_centralcore = contains(CH_AA, CH_list(mcf_channel));
        CH_AA = CH_AA(~mask_centralcore);
        
    catch
        disp("Not an MCF Needle!");
    end
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

%% Consolidate the readings into one W data matrix
% get the actual curvature we are trying to fit
calib_mask = strcmp(data_mats.AA1.('type'), 'calibration');
valid_mask = strcmp(data_mats.AA1.('type'), 'validation');
mask = calib_mask | valid_mask;
W_act_aai = data_mats.(AA_i){mask, ["Curvature_x (1/m)", "Curvature_y (1/m)"]};
w_act = reshape(W_act_aai', [], 1);

% get the W data matrix
W = zeros(length(w_act), length(AA_list)); % the measurement array
for i = 1:length(AA_list)
    AA_i = AA_list(i);
    CH_AA_i = CH_AA(contains(CH_AA, AA_i));
    
    % get the calibration matrix and signals
    cal_mat = double(fbg_needle.aa_cal(AA_i).T);
    signals = data_mats.(AA_i){mask,CH_AA_i};
    
    % compute the data matrices for this AA
    W_meas_aai = signals * cal_mat;
%     W_meas_aai = data_mats.(AA_i){mask, AA_i +[ " Predicted Curvature_x (1/m)", " Predicted Curvature_y (1/m)"]};
    
    % add it to the measurement array
    W(:, i) = reshape(W_meas_aai', [], 1);
    
    
end

% % append the 'sum(weights) = 1' constraint ( eta := weights)
% W = [W; ones(1, size(W, 2))];
% w_act = [w_act; 1];

%% perform the non-negative least squares fit
curv_weight = curv_weight_rule(abs(w_act),~curv_weighting);
D = sqrt(diag(curv_weight));
% D = D./size(D, 1);

% equal weighting here
A_eq_wgt = equal_weighting_lambda * eye(length(AA_list));
b_eq_wgt = equal_weighting_lambda * ones(length(AA_list),1)/length(AA_list);

A = [D*W; A_eq_wgt];
b = [D*w_act; b_eq_wgt]; 
weights = lsqlin(A, b, [], [], ones(1, size(W, 2)), [1], [0, 0, 0.075, 0.025], []); % set lower bound
weights = weights./sum(weights); % normalize just in case
disp(" ");

% diaplay results
if curv_weighting
    disp("Curvature weighting: Clinically Relevant");
else
    disp("Curvature weighting: Equal");
end

fprintf("Equal Weigting lambda: %f\n", equal_weighting_lambda);
disp(" ");

disp("Weights:");
disp([AA_list; reshape(weights, 1, [])] )
base_print = ['[ ', repmat('%f, ', 1, size(W, 2) - 1), '%f ]\n\n'];
fprintf(base_print, weights);

disp('Total error');
error = norm((W * weights - w_act));
disp(error)
disp("Mean Error");
disp(2*error/size(W,1));

%% save the weights to the fbg_needle 
py_dict_weights = py.dict();
for i=1:length(AA_list)
    py_dict_weights{AA_list(i)} = weights(i);
    
end
if save_bool
    fbg_needle.set_weights(py_dict_weights);
    fbg_needle.save_json(fbgneedle_param_weight);
    fprintf('Saved json parmater file: %s\n', fbgneedle_param_weight);
end
%% Functions
% function for performing weighted least squares in curvature
function curv_weight = curv_weight_rule(k, trivial)
    disp(nargin)
    curv_weight = ones(length(k), 1);
    
    if ~trivial 
        for i = 1:length(k)
            if abs(k(i)) <= 1
                curv_weight(i) = 1;

            else
                curv_weight(i) = 0.05;

            end
            
            if mod(i, 2) == 0
                angle = atan2(k(i), k(i-1));
                if rad2deg(angle) == 90 
                    curv_weight(i) = 1;
                    curv_weight(i-1) = 1;
                end
            end
        end
    end
end

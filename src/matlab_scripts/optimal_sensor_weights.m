%% optimal_sensor_weights.m
%
% this is a script to determine the optimal sensor weighting for the 
% FBG sensors by solving a least squares problem
%
% - written by: Dimitri Lezcano

%% Set-up 
% options
save_bool = false;

% python set-up
if ispc % windows file system
    pydir = "..\";
    
else
    pydir = "../";
    
end

if count(py.sys.path, pydir) == 0
    insert(py.sys.path, int32(0), pydir);
end

% file set-up
directory = "../../FBG_Needle_Calibration_Data/needle_3CH_4AA/";
fbgneedle_param = directory + "needle_params-Jig_Calibration_08-05-20.json";
fbgneedle_param_out = strrep(fbgneedle_param, '.json', '_weighted.json');

datadir = directory + "Jig_Calibration_08-05-20/"; % calibration data
% datadir = directory + "Validation_Jig_Calibration_08-19-20/"; % validation data
data_mats_file = datadir + "Data Matrices_new.xlsx";
data_mats_proc_file = strrep(data_mats_file, '.xlsx', '_proc.xlsx');
fig_save_file = datadir + "Jig_Shape_fit";

% paramteter set-up
jig_offset = 26.0; % the jig offset of full insertion
AA_weights = [];% [1, 0.9, 0.3, 0.0]; % [AA1, AA2, AA3, AA4] reliability weighting
if ~isempty(AA_weights)
    fig_save_file = fig_save_file + "_weighted";
    
end

%% Load FBGNeedle python class
fbg_needle = py.FBGNeedle.FBGNeedle.load_json(fbgneedle_param);
disp("FBGNeedle class loaded.")

% channel list
ch_list = 1:double(fbg_needle.num_channels);
CH_list = "CH" + ch_list;
aa_list = 1:double(fbg_needle.num_aa);
AA_list = "AA" + aa_list; % the "AAX" string version

%% load the processed data matrices
data_mats = struct();
for AA_i = AA_list
    data_mats.(AA_i) = readtable(data_mats_proc_file, 'Sheet', AA_i, ...
        'PreserveVariableNames', false, 'ReadRowNames', true); % remove the first column (exp #)
    disp(AA_i + " loaded.");
end
disp(' ');

%% Consolidate the readings into one W data matrix
% get the actual curvature we are trying to fit
W_act_aai = data_mats.(AA_i)(:, ["CurvatureX", "CurvatureY"]).Variables;
w_act = reshape(W_act_aai', [], 1);

% get the W data matrix
W = zeros(length(w_act), length(AA_list)); % the measurement array
for i = 1:length(AA_list)
    AA_i = AA_list(i);
    
    % get the data matrices for this AA
    W_meas_aai = data_mats.(AA_i)(:, ["PredCurvX", "PredCurvY"]).Variables;
    
    % add it to the measurement array
    W(:, i) = reshape(W_meas_aai', [], 1);
    
    
end

% append the 'sum(weights) = 1' constraint ( eta := weights)
W = [W; ones(1, size(W, 2))];
w_act = [w_act; 1];

%% perform the non-negative least squares fit
weights = lsqnonneg(W, w_act);
weights = weights./sum(weights); % normalize just in case

disp("Weights:");
disp([AA_list; reshape(weights, 1, [])] )

disp('Total error');
disp(norm((W * weights - w_act)))

%% save the weights to the fbg_needle 
py_dict_weights = py.dict();
for i=1:length(AA_list)
    py_dict_weights{AA_list(i)} = weights(i);
    
end

fbg_needle.set_weights(py_dict_weights);
fbg_needle.save_json(fbgneedle_param_out);
fprintf('Saved json parmater file: %s\n', fbgneedle_param_out);



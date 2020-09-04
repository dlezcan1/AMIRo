%% jig_shape_validation.m
%
% this is a script to perform shape validation using the jig from an fbg needle
%
% - written by: Dimitri Lezcano

%% Set-up
% python set-up
pydir = "..\";
if count(py.sys.path, pydir) == 0
    insert(py.sys.path, int32(0), pydir);
end

% file set-up
directory = "../../FBG_Needle_Calibration_Data/needle_3CH_4AA/";
fbgneedle_param = directory + "needle_params-Jig_Calibration_08-05-20.json";

datadir = directory + "Validation_Jig_Calibration_08-19-20/";
data_mats_file = datadir + "Data Matrices.xlsx";
data_mats_proc_file = strrep(data_mats_file, '.xlsx', '_proc.xlsx');

%% Load FBGNeedle python class
fbg_needle = py.FBGNeedle.FBGNeedle.load_json(fbgneedle_param);
disp("FBGNeedle class loaded.")

% channel list
ch_list = 1:double(fbg_needle.num_channels);
CH_list = "CH" + ch_list;
aa_list = 1:double(fbg_needle.num_aa);
AA_list = "AA" + aa_list; % the "AAX" string version

%% load the data matrices
data_mats = struct();
for AA_i = AA_list
    data_mats.(AA_i) = readtable(data_mats_file, 'Sheet', AA_i, ...
        'PreserveVariableNames', false, 'ReadRowNames', true); % remove the first column (exp #)
    disp(AA_i + " loaded.");
end
disp(' ');

%% determine the calculated curvature for each AA
for AA_i = AA_list
    cal_mat = double(fbg_needle.aa_cal(AA_i).T);
    signal_mat = data_mats.(AA_i)(:, CH_list).Variables;
    curvature_mat = signal_mat * cal_mat;
    data_mats.(AA_i).PredCurvX = curvature_mat(:,1);
    data_mats.(AA_i).PredCurvY = curvature_mat(:,2);
    data_mats.(AA_i).PredCurv = vecnorm(curvature_mat, 2, 2);
    disp("Calculated predicted curvature vector from " + AA_i + ".");
end
disp(" ");

%% Save the data as a processed data matrices file
for AA_i = AA_list
    writetable(data_mats.(AA_i),data_mats_proc_file, 'Sheet', AA_i, ...
        'WriteRowNames', true);
    
end

fprintf("Wrote processed data file: %s\n\n", data_mats_proc_file);

%% Functions 
% TODO: jig (constant curvature) shape model
% TODO: error in positions as a function of arclength (s)
% TODO: error in positions as a function of z-position (z)
%           - will need some sort of interpolation


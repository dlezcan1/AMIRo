%% process_insertionval.m
%
% this is a script to run through the data points and generate the FBG shape
% from measurements
% 
% - written by: Dimitri Lezcano

%% Set-up
% directories to iterate through
trial_dirs = dir("../../data/01-18-2021_Test-Insertion-Expmt/Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% files to find
fbgdata_file = "FBGdata_meanshift.xls";

% saving options
save_bool = true;
fbgout_posfile = "FBGdata_3d-position.xls";
fbgout_paramfile = "FBGdata_3d-params.txt";

% directory separation
if ispc
    dir_sep = '\';
else
    dir_sep = '/';
end

% calibraiton matrices file
calib_dir = "../../data/01-18-2021_Test-Insertion-Expmt/";
calib_file = calib_dir + "needle_params-Jig_Calibration_11-15-20_weighted.json";

% Initial guesses for kc and w_init
kc_i = 0.002;
w_init_i = [kc_i; 0; 0]; % ideal insertion
theta0 = 0;


%% Load the calibration matrices and AA locations (from base)
fbgneedle = jsondecode(fileread(calib_file));

% AA parsing
num_aas = fbgneedle.x_ActiveAreas;
aa_base_locs_tot = struct2array(fbgneedle.SensorLocations); % total insertion length
aa_tip_locs = fbgneedle.length - aa_base_locs_tot;
cal_mats_cell = struct2cell(fbgneedle.CalibrationMatrices);
cal_mat_tensor = cat(3, cal_mats_cell{:});


%% Iterate through the files
f3d = figure(1);
set(f3d,'units','normalized','position', [0, 0.4, 1/3, .5])
f2d = figure(2);
set(f2d,'units','normalized','position', [1/3, 0.4, 1/3, .5] )
for i = 1:length(trial_dirs)
    tic; 
    % trial operations
    L = str2double(trial_dirs(i).name);
    
    % trial directory
    d = strcat(trial_dirs(i).folder,dir_sep, trial_dirs(i).name, dir_sep);
    fbg_file = d + fbgdata_file;
    
    % load the fbg shift in
    wl_shift = readmatrix(fbg_file);
    wl_shift = reshape(wl_shift, [], 3)'; % reshape the array so AA's are across rows and Ch down columns
    
    % use calibration senssors
    curvatures = calibrate_fbgsensors(wl_shift, cal_mat_tensor);
        
    % get the shape
    [pos, wv, Rmat, kc, w_init] = singlebend_needleshape(curvatures, aa_tip_locs, L, kc_i, w_init_i, theta0);
    t = toc;
    
    % set new predictions
    kc_i = kc; 
    if i == 1
        w_init_i = w_init;
    elseif i > 1 && strcmp(trial_dirs(i).folder, trial_dirs(i-1).folder) 
        w_init_i = w_init;
    else
        w_init_i = [kc_i; 0; 0];
    end
    
    % plotting
    %- 3D
    figure(1);
    plot3(pos(3,:), pos(1,:), pos(2,:), 'linewidth', 2);
    axis equal; grid on;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold'); 
    zlabel('y [mm]', 'FontWeight', 'bold');
    title(d);
    
    %- 2D
    figure(2);
    subplot(2,1,1);
    plot(pos(3,:), pos(2,:), 'LineWidth', 2);
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    subplot(2,1,2);
    plot(pos(3,:), pos(1,:), 'LineWidth', 2);
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    % save the data
    if save_bool
       % write position file
       writematrix(pos, d + fbgout_posfile);
       fprintf("Wrote 3D Positions: '%s'\n", d + fbgout_posfile);
       
       % write shape sensing parameters
       T = table(kc, w_init', 'VariableNames', {'kc', 'w_init'});
       writetable(T, d + fbgout_paramfile);
       fprintf("Wrote 3D Position Params: '%s'\n", d + fbgout_paramfile);
       
    end
    
    % output
    fprintf("Finished trial: '%s' in %.2f secs.\n", d, t);
    disp(" ");
end

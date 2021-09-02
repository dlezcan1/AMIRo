
%% process_insertionval.m
%
% this is a script to run through the data points and generate the FBG shape
% from measurements
% 
% - written by: Dimitri Lezcano
configure_env on;

%% Set-up
% directories to iterate through
expmt_dir = "../../data/3CH-4AA-0004/08-30-2021_Insertion-Expmt-1/";
trial_dirs = dir(expmt_dir + "Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% files to find
fbgdata_file = "FBGdata.xls";
fbgdata_ref_wl_file = fullfile(expmt_dir, "Reference/Insertion1/0", fbgdata_file);
ref_wl_per_trial = ~isfile(fbgdata_ref_wl_file); % try not to use where possible

% weighted FBG measurement options
use_weights = true;

% python FBGNeedle class usage
python_fbgneedle = true;

% saving options
save_bool = true;
fbgout_basefile = "FBGdata";
if use_weights == true
    fbgout_basefile = fbgout_basefile + "_FBG-weights";
end    
fbgout_posfile = fbgout_basefile + "_3d-position.xls";
fbgout_paramfile = fbgout_basefile + "_3d-params.txt";


% directory separation
if ispc
    dir_sep = '\';
else
    dir_sep = '/';
end

% calibraiton matrices file
calib_dir = "../../data/3CH-4AA-0004/";
calib_file = fullfile(calib_dir, "needle_params_08-16-2021_Jig-Calibration_best.json");

% Initial guesses for kc and w_init
kc_i = 0.002;
w_init_i = [kc_i; 0; 0]; % ideal insertion
theta0 = 0;

%% Load the reference wavelengths
if ~ref_wl_per_trial % only one reference wavelength
    ref_wls_mat = readmatrix(fbgdata_ref_wl_file, 'sheet', 'Sheet1');
    ref_wls = mean(ref_wls_mat, 1); % the reference wavelengths
end

%% Load the calibration matrices and AA locations (from base)
if python_fbgneedle
    fbgneedle = py.sensorized_needles.FBGNeedle.load_json(calib_file);
    
    num_chs = double(fbgneedle.num_channels);
    num_aas = double(fbgneedle.num_activeAreas);
    aa_base_locs_tot = double(py.numpy.array(fbgneedle.sensor_location));
    aa_tip_locs = fbgneedle.length - aa_base_locs_tot;
    
    cal_mat_tensor = zeros(2, num_chs, num_aas);
    weights = zeros(1, num_aas);
    for i = 1:num_aas
        AAi = sprintf("AA%d", i);
        cal_mat_aai = double(fbgneedle.aa_cal(AAi));
        cal_mat_tensor(:,:,i) = cal_mat_aai;
        
        weights(i) = double(fbgneedle.weights{fbgneedle.aa_loc(AAi)});
    end
    
else
    fbgneedle = jsondecode(fileread(calib_file));

    % AA parsing
    num_aas = fbgneedle.x_ActiveAreas;
    aa_base_locs_tot = struct2array(fbgneedle.SensorLocations); % total insertion length
    weights = struct2array(fbgneedle.weights); 
    aa_tip_locs = fbgneedle.length - aa_base_locs_tot;
    cal_mats_cell = struct2cell(fbgneedle.CalibrationMatrices);
    cal_mat_tensor = cat(3, cal_mats_cell{:});
end

if all(weights == 0)
    weights = ones(size(weights));
end

%% Iterate through the files
lshift = 1/6;
f3d = figure(1);
set(f3d,'units','normalized','position', [lshift + 0, 0.5, 1/3, .42]);

f2d = figure(2);
set(f2d,'units','normalized','position', [lshift + 1/3, 0.5, 1/3, .42] );

f3d_insert = figure(3);
set(f3d_insert, 'units', 'normalized', 'position', [0, 0, 1/3, 0.42]);

f2d_insert = figure(4);
set(f2d_insert, 'units', 'normalized', 'position', [2/3, 0, 1/3, 0.42]); 

fkc = figure(5);
set(fkc, 'units', 'normalized', 'position', [1/3, 0, 1/3, 0.42]);
    
dir_prev = "";
kc_vals = [];
depths = [];
for i = 1:length(trial_dirs)
    if (~strcmp(dir_prev, trial_dirs(i).folder) && ~strcmp(dir_prev, "")) 
        % plot depths vs kc
        figure(5);
        plot(depths, kc_vals, '*-');
        xlabel("Insertion Depth (mm)"); ylabel("\kappa_c (1/mm)");
        title(sprintf("Insertion #%d", hole_num) + " | \kappa_c vs Insertion Depth");
        
        % save the figure
        if save_bool
           saveas(fkc, fileout_base + "_kc-all-insertions.png");
           fprintf("Saved figure #d: '%s'\n", fkc.Number, fileout_base + "_kc-all-insertions.png");
        end
        
        % empty the values
        kc_vals = [];
        depths = [];
    end
    
    tic; 
    % trial operations
    L = str2double(trial_dirs(i).name);
    re_ret = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
    hole_num = str2double(re_ret{1}{1});
    ins_depth = str2double(trial_dirs(i).name);
    
    % trial directory
    d = fullfile(trial_dirs(i).folder, trial_dirs(i).name);
    fbg_file = fullfile(d, fbgdata_file);
    
    % load the fbg shift in
    wls_mat = readmatrix(fbg_file, 'Sheet', 'Sheet1');
    wls_mat = wls_mat(all(wls_mat > 0, 2), :); % filter out any rows w/ 0 as FBG signal
    wls_mean = mean(wls_mat, 1);
    if ref_wl_per_trial
        ref_wls_mat = readmatrix(fullfile(trial_dirs(i).folder, '0', fbgdata_file));
        ref_wls_mat = ref_wls_mat(all(ref_wls_mat > 0, 2), :);
        ref_wls     = mean(ref_wls_mat, 1);
    end
    
    wls_shift = wls_mean - ref_wls;
    wls_shift = reshape(wls_shift, [], num_chs)'; % reshape the array so AA's are across rows and Ch down columns
    
    % apply temperature compensation
    wl_shift_Tcorr = temperature_compensate(wls_shift);
    
    % use calibration senssors
    curvatures = calibrate_fbgsensors(wl_shift_Tcorr, cal_mat_tensor);
        
    % get the shape
    [pos, wv, Rmat, kc, w_init] = singlebend_needleshape(curvatures, aa_tip_locs, L, kc_i, w_init_i, theta0, weights);
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
    
    % total insertion plots
    % - 3D total
    figure(3);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot3(pos(3,:), pos(2,:), pos(1,:), 'linewidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold'); 
    zlabel('y [mm]', 'FontWeight', 'bold');
    legend();  
    axis equal; grid on;
    title(sprintf("Insertion #%d | FBG Shape Determination", hole_num));
    
    % - 3D total
    figure(4);
    subplot(2,1,1);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(pos(3,:), pos(2,:), 'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    subplot(2,1,2);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(pos(3,:), pos(1,:), 'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    legend()
    sgtitle(sprintf("Insertion #%d | FBG Shape Determination", hole_num));
    
    
    
    % add to kappa c records
    kc_vals = [kc_vals, kc];
    depths = [depths; ins_depth];
    
    % save the data
    if save_bool
       % write position file
       writematrix(pos, fullfile(d, fbgout_posfile));
       fprintf("Wrote 3D Positions: '%s'\n", fullfile(d, fbgout_posfile));
       
       % write shape sensing parameters
       T = table(kc, w_init', theta0, L, 'VariableNames', ...
           {'kc', 'w_init', 'theta0', 'L'});
       writetable(T, fullfile(d, fbgout_paramfile));
       fprintf("Wrote 3D Position Params: '%s'\n", fullfile(d, fbgout_paramfile));
       
       % save figures
       fileout_base = fullfile(trial_dirs(i).folder, fbgout_basefile);
       saveas(f3d_insert, fileout_base + "_3d-all-insertions.png");
       fprintf("Saved figure #%d: '%s'\n", f3d_insert.Number, ...
           fileout_base + "_3d-all-insertions.png");
       
       saveas(f2d_insert, fileout_base + "_2d-all-insertions.png");
       fprintf("Saved figure #%d: '%s'\n", f2d_insert.Number, ...
           fileout_base + "_2d-all-insertions.png");
       
    end
    
    % update previous directory
    dir_prev = trial_dirs(i).folder;
    
    if i == length(trial_dirs) % handle the last edge case
        % plot depths vs kc
        figure(5);
        plot(depths, kc_vals, '*-');
        xlabel("Insertion Depth (mm)"); ylabel("\kappa_c (1/mm)");
        title(sprintf("Insertion #%d", hole_num) + " | \kappa_c vs Insertion Depth");
        
        % save the figure
        if save_bool
           saveas(fkc, fileout_base + "_kc-all-insertions.png");
           fprintf("Saved figure #d: '%s'\n", fkc.Number, fileout_base + "_kc-all-insertions.png");
        end
        
        % empty the values
        kc_vals = [];
        depths = [];
    end
    
    % output
    fprintf("Finished trial: '%s' in %.2f secs.\n", d, t);
    disp(" ");
    
    
end

%% Completion
close all;
disp("Program Terminated.");


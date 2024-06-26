%% process_insertionval_2layer.m
%
% this is a script to run through the data points and generate the FBG shape
% from measurements using double-layer shape sensing
% 
% - written by: Dimitri Lezcano
configure_env on;
clear;
%% Set-up
% directories to iterate through
expmt_dir = "../../data/3CH-4AA-0004/2021-11-10_Insertion-Expmt-1/"; % CAN CHANGE
trial_dirs = dir(expmt_dir + "Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories
expmt_paths = split(strip(fullfile(expmt_dir), filesep), filesep);
expmt_subdir = expmt_paths(end);

% files to find
fbgdata_file = "FBGdata.xls";
fbgdata_ref_wl_file = fullfile(expmt_dir, "Reference/Insertion1/125", fbgdata_file);
ref_wl_per_trial = ~isfile(fbgdata_ref_wl_file); % try not to use where possible

% weighted FBG measurement options
use_weights = true; % CAN CHANGE, BUT PROBABLY KEEP "true"

% python FBGNeedle class usage
python_fbgneedle = false; % CAN CHANGE, BUT PROBABLY KEEP "true"

% saving options
save_bool = true;  % CAN CHANGE 
fbgout_basefile = "FBGdata_2layer";
if use_weights == true
    fbgout_basefile = fbgout_basefile + "_FBG-weights";
end    
fbgout_posfile = fbgout_basefile + "_3d-position.xls";
fbgout_paramfile = fbgout_basefile + "_3d-params.txt";

% calibration matrices matrices file: CAN CHANGE PER NEEDLE
needle_dir = "../../data/3CH-4AA-0004/"; % the needle you are using
needle_calib_file = fullfile(needle_dir, "needle_params_2021-08-16_Jig-Calibration_best.json");

% needle mechanical properties
needle_gauge = 18;
needle_mechparam_file = fullfile('../../shape-sensing', ...
    sprintf('shapesensing_needle_properties_%dG.mat', needle_gauge));
needle_mechparams = load(needle_mechparam_file);

% Initial guesses for kc and w_init DON'T CHANGE
kc1_i = 0.003; % hard
kc2_i = 0.002; % soft
w_init_i = [kc1_i; 0; 0]; % ideal insertion
theta0 = 0;

%% Read in the experiment.json file to find the 2nd layer length
experiment_description = jsondecode(fileread(fullfile(expmt_dir, 'experiment.json')));
z_crit = experiment_description.tissue1Length;

%% Load the calibration matrices and AA locations (from base) DON'T CHANGE
if python_fbgneedle
    fbgneedle = py.sensorized_needles.FBGNeedle.load_json(needle_calib_file);
    
    num_chs = double(fbgneedle.num_channels);
    num_aas = double(fbgneedle.num_activeAreas);
    aa_base_locs_tot = double(py.numpy.array(fbgneedle.sensor_location));
    aa_tip_locs = fbgneedle.length - aa_base_locs_tot;
    
    chaa_all = fbgneedle.generate_chaa();
    chaa = cellfun(@(x) string(x), cell(chaa_all{1}));
    
    cal_mat_tensor = zeros(2, num_chs, num_aas);
    weights = zeros(1, num_aas);
    for i = 1:num_aas
        AAi = sprintf("AA%d", i);
        cal_mat_aai = double(fbgneedle.aa_cal(AAi));
        cal_mat_tensor(:,:,i) = cal_mat_aai;
        
        weights(i) = double(fbgneedle.weights{fbgneedle.aa_loc(AAi)});
    end
    
else
    fbgneedle = jsondecode(fileread(needle_calib_file));

    % AA parsing
    num_chs = fbgneedle.x_Channels;
    num_aas = fbgneedle.x_ActiveAreas;
    aa_base_locs_tot = struct2array(fbgneedle.SensorLocations); % total insertion length
    weights = struct2array(fbgneedle.weights); 
    aa_tip_locs = fbgneedle.length - aa_base_locs_tot;
    cal_mats_cell = struct2cell(fbgneedle.CalibrationMatrices);
    cal_mat_tensor = cat(3, cal_mats_cell{:});%.*[1;-1];
    
    chaa = split(sprintf("CH%d | AA%d,", cart_product((1:num_chs)', (1:num_aas)')'), ',')';
    chaa = chaa(1:end-1);
    
end

if all(weights == 0)
    weights = ones(size(weights));
end

%% Load the reference wavelengths
if ~ref_wl_per_trial % only one reference wavelength
    ref_wls_mat = readmatrix(fbgdata_ref_wl_file, 'sheet', 'Sheet1');
    ref_wls_mat = ref_wls_mat(all(ref_wls_mat > 0,2),:);  % remove all errors
    ref_wls = mean(ref_wls_mat, 1); % the reference wavelengths
end

%% Set-up the figures
lshift = 0;
f3d = figure(1);
set(f3d,'units','normalized','position', [lshift + 0, 0.5, 1/3, .42]);

f2d = figure(2);
set(f2d,'units','normalized','position', [lshift + 1/3, 0.5, 1/3, .42] );

f3d_insert = figure(3);
set(f3d_insert, 'units', 'normalized', 'position', [lshift + 0, 0, 1/3, 0.42]);

f2d_insert = figure(4);
set(f2d_insert, 'units', 'normalized', 'position', [lshift + 2/3, 0, 1/3, 0.42]); 

fwl_dist = figure(5);
set(fwl_dist, 'units', 'normalized', 'position', [1 + 1/16, 0.05,1,0.875]);


%% Set-up the results table
chaa_cols = reshape(strrep(chaa, ' ', ''), 1, []);
curv_cols = reshape(splitlines(strip(sprintf("Curv_%s AA%s\n", cart_product(['x','y']', ...
                        string([1:num_aas])')'), 'right')), 1, []);
%%%%%%%%%%%%%%%%%%%ALEX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
kc_w_init_final_name = ["Insertion Hole","Insertion Depth", "Num Layers", "kc1","kc2",...
    "w_init_1","w_init_2","w_init_3",...
    chaa_cols, curv_cols, strcat("FBG ", chaa_cols)];
kc_w_init_final_type = ['uint8', repmat("double", 1, numel(kc_w_init_final_name)-1)];
kc_w_init_final_tbl = table('size',[0,numel(kc_w_init_final_name)],'VariableTypes',kc_w_init_final_type...
    ,'VariableNames',kc_w_init_final_name);
%%%%%%%%%%%%%%%%%%%ALEX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Iterate through the trials
dir_prev = "";
for i = 1:length(trial_dirs)
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
        
    % get the shape (single-layer)
    [pos, wv, Rmat, kc1, kc2, w_init, s_crit] = singlebend_doublelayer_needleshape(curvatures, aa_tip_locs,...
        needle_mechparams, L, z_crit, kc1_i, kc2_i, w_init_i, theta0, weights);
    t = toc;
    num_layers = 2 + min(0, sign(kc2)); % subtract a layer if kc2 < 0
    
    % set new predictions
    kc1_i = kc1;
    kc2_i = kc2;
    if i == 1
        w_init_i = w_init;
    elseif i > 1 && strcmp(trial_dirs(i).folder, trial_dirs(i-1).folder) 
        w_init_i = w_init;
    else
        w_init_i = [kc1_i; 0; 0];
    end
    
    % plotting
    [~, min_z_idx] = min(abs(pos(3,:) - z_crit));
    xc = pos(1,min_z_idx); yc = pos(2,min_z_idx);
        
    %- 3D
    figure(f3d);
    plot3(pos(3,:), pos(2,:), pos(1,:), 'linewidth', 2); hold on;
    plot_tissueboundary3d(z_crit, yc, xc, 5, 5); hold off;
    axis equal; grid on;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold'); 
    zlabel('x [mm]', 'FontWeight', 'bold');
    title(d);
    
    %- 2D
    figure(f2d);
    subplot(2,1,1);
    plot(pos(3,:), pos(2,:), 'LineWidth', 2); 
    xline(z_crit, 'r--', 'Tissue Boundary');
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    subplot(2,1,2);
    plot(pos(3,:), pos(1,:), 'LineWidth', 2); 
    xline(z_crit, 'r--', 'Tissue Boundary'); 
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    % total insertion plots
    % - 3D total
    figure(f3d_insert);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot3(pos(3,:), pos(2,:), pos(1,:), 'linewidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    if ~strcmp(dir_prev, trial_dirs(i).folder)
        plot_tissueboundary3d(z_crit, yc, xc, 5, 5); hold on;
    end
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold'); 
    zlabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    title(sprintf("Insertion #%d | FBG Shape Determination", hole_num));
    
    % - 2D total
    figure(f2d_insert);
    subplot(2,1,1);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(pos(3,:), pos(2,:), 'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        xline(z_crit, 'r--', 'Tissue Boundary');
    end
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    subplot(2,1,2);
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        hold off;
    end
    plot(pos(3,:), pos(1,:), 'LineWidth', 2, 'DisplayName', sprintf("%.1f mm", L)); hold on;
    if ~strcmp(dir_prev, trial_dirs(i).folder) % new trial
        xline(z_crit, 'r--', 'Tissue Boundary');
    end
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    sgtitle(sprintf("Insertion #%d | FBG Shape Determination", hole_num));
    
    % signals
    figure(fwl_dist);
    for chaa_i = 1:num_aas * num_chs
        CH_i_AA_j = chaa(chaa_i);
        subplot(num_chs, num_aas,chaa_i);
        histogram(wls_mat(:,chaa_i));
        xlabel('Wavelength (nm)');
        title(CH_i_AA_j);
    end
    
    % save the data
    if save_bool
       % write position file
       writematrix(pos, fullfile(d, fbgout_posfile));
       fprintf("Wrote 3D Positions: '%s'\n", fullfile(d, fbgout_posfile));
       
       % write shape sensing parameters
       T = table(kc1, kc2, w_init', theta0, L, s_crit, 'VariableNames', ...
           {'kc1', 'kc2', 'w_init', 'theta0', 'L', 's_crit'});
       writetable(T, fullfile(d, fbgout_paramfile));
       fprintf("Wrote 3D Position Params: '%s'\n", fullfile(d, fbgout_paramfile));
       
       % save figures
       fileout_base = fullfile(trial_dirs(i).folder, fbgout_basefile);
       
       savefigas(f3d_insert, strcat(fileout_base, "_3d-all-insertions"), 'Verbose', true);
       
       savefigas(f2d_insert, strcat(fileout_base, '_2d-all-insertions'), 'Verbose', true);
       
       savefigas(fwl_dist, strcat(fileout_base, '_peak-distribution'), 'Verbose', true);
       
    end
    
    % update previous directory
    dir_prev = trial_dirs(i).folder;

    %%%%%%%%%%%%%%%ALEX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fbg_data = cell(1,size(wls_mat,2));
    for chaa_i = 1:size(wls_mean, 2)
        fbg_data{chaa_i} = wls_mat(:,chaa_i);
    end
    
    %appending result to kc/w_init table
    kc_w_init_final_tbl = [kc_w_init_final_tbl; 
        {hole_num,ins_depth, num_layers, kc1, kc2,w_init(1),w_init(2),w_init(3),...
        wl_shift_Tcorr(1,1),wl_shift_Tcorr(1,2),wl_shift_Tcorr(1,3),wl_shift_Tcorr(1,4),...
        wl_shift_Tcorr(2,1),wl_shift_Tcorr(2,2),wl_shift_Tcorr(2,3),wl_shift_Tcorr(2,4),...
        wl_shift_Tcorr(3,1),wl_shift_Tcorr(3,2),wl_shift_Tcorr(3,3),wl_shift_Tcorr(3,4),...
        curvatures(1,1),curvatures(2,1),curvatures(1,2),curvatures(2,2),...
        curvatures(1,3),curvatures(2,3),curvatures(1,4),curvatures(2,4),...
        fbg_data{1},fbg_data{2},fbg_data{3},fbg_data{4},...
        fbg_data{5},fbg_data{6},fbg_data{7},fbg_data{8},...
        fbg_data{9},fbg_data{10},fbg_data{11},fbg_data{12}}];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % output
    fprintf("Finished trial: '%s' in %.2f secs.\n", d, t);
    disp(" ");
    
end

final_table_sorted = sortrows(kc_w_init_final_tbl, [1,2]);

%% Save the table
if save_bool
    fbgout_results_file = fullfile(expmt_dir, strcat(fbgout_basefile, '_results'));
    
    save(strcat(fbgout_results_file, '.mat'), 'final_table_sorted' );
    fprintf("Saved results table to: %s\n", strcat(fbgout_results_file, '.mat'));
    
    include_cols = ~contains(final_table_sorted.Properties.VariableNames,"FBG CH");
    writetable(final_table_sorted(:,include_cols), strcat(fbgout_results_file, '.xlsx'));
    fprintf("Saved results table to: %s\n", strcat(fbgout_results_file, '.xlsx'));
    
end

%% Plotting from table
fig_table_wls = figure(9);
set(fig_table_wls, 'units', 'normalized', 'position', [0.1, 0.1,0.8,0.8]);
fig_table_kc = figure(10);
fig_table_kc_win = figure(11);
fig_table_winit = figure(12);
fig_table_curv = figure(13);
set(fig_table_curv, 'units', 'normalized', 'position', [1 + 0.15, 0.1,0.8,0.8]);

for i = unique(final_table_sorted.('Insertion Hole'))'
    ii = i-1;Nprev = 0;
    index = int2str(i);
    table_fig_save = fullfile(expmt_dir, sprintf("Insertion%d/", i), fbgout_basefile);
    insertion_mask = final_table_sorted.('Insertion Hole') == i;
    depth_mask = final_table_sorted.('Insertion Depth') > 30;
    kc2_mask = final_table_sorted.kc2 >= 0;
    
    % plot wls_shift_tcorr for channels/AA
    figure(fig_table_wls)
    hold off;
    for aa_k = 1:num_aas
        CHAA_cols = strcat("CH", string(1:3), "|AA", string(aa_k));
        
        subplot(2,2,aa_k)
        hold off;
        plot(final_table_sorted{insertion_mask, 'Insertion Depth'}, ...
             final_table_sorted{insertion_mask, CHAA_cols(1)}, '.-'); hold on;
        plot(final_table_sorted{insertion_mask, 'Insertion Depth'}, ...
             final_table_sorted{insertion_mask, CHAA_cols(2)}, '.-'); hold on;
        plot(final_table_sorted{insertion_mask, 'Insertion Depth'}, ...
             final_table_sorted{insertion_mask, CHAA_cols(3)},' .-'); hold on;
        xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold on;
        
        title(sprintf("AA%d",aa_k));
        xlabel("Insertion Depth (mm)");ylabel("Wavelength Shift (nm)");
        if aa_k == num_aas
            l = legend("CH1","CH2","CH3",'Fontsize',10, 'Orientation', 'horizontal');
            l.Position(1:2) = [(1 - l.Position(3))/2, 0.02];
        end
    end
    sgtitle(sprintf("Insertion #%d | Wavelength shift", i));
    if save_bool
        savefigas(fig_table_wls, strcat(table_fig_save, "_table_wls_shift"));
    end
        
    % plot kc
    figure(fig_table_kc);
    hold off;
    plot(final_table_sorted{insertion_mask, 'Insertion Depth'}, ...
         final_table_sorted{insertion_mask, 'kc1'}, '*-', 'DisplayName', '\kappa_{c,1}'); hold on;
    plot(final_table_sorted{insertion_mask & kc2_mask, 'Insertion Depth'}, ...
         final_table_sorted{insertion_mask & kc2_mask, 'kc2'}, '*-', 'DisplayName', '\kappa_{c,2}'); hold on;
    xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold on;
    legend('location', 'northeastoutside')
    title(strcat(sprintf("Insertion #%d | ", i), "\kappa_c vs Insertion Depth"));
    
    xlabel("Insertion Depth");ylabel("\kappa_c (1/mm)");
    if save_bool
        savefigas(fig_table_kc, strcat(table_fig_save, "_table_kc.png"));
    end
    
    % plot kc windowed
    figure(fig_table_kc_win);
    hold off;
    plot(final_table_sorted{insertion_mask & depth_mask, 'Insertion Depth'}, ...
         final_table_sorted{insertion_mask & depth_mask, 'kc1'}, '*-', ...
         'DisplayName', '\kappa_{c,1}'); hold on;
     plot(final_table_sorted{insertion_mask & kc2_mask & depth_mask, 'Insertion Depth'}, ...
          final_table_sorted{insertion_mask & kc2_mask & depth_mask, 'kc2'}, '*-', ...
          'DisplayName', '\kappa_{c,2}'); hold on;
    xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); 
    legend('location', 'northeastoutside')
    title(strcat(sprintf("Insertion #%d | ", i), "\kappa_c vs Insertion Depth Windowed"));
    
    % plot w_init1,2,3
    figure(fig_table_winit);
    hold off;
    w_init_cols = strcat('w_init_', string(1:3));
    plot(final_table_sorted{insertion_mask, 'Insertion Depth'},...
         final_table_sorted{insertion_mask, w_init_cols(1)}, '.-'); hold on;
    plot(final_table_sorted{insertion_mask, 'Insertion Depth'},...
         final_table_sorted{insertion_mask, w_init_cols(2)}, '.-'); hold on;
    plot(final_table_sorted{insertion_mask, 'Insertion Depth'},...
         final_table_sorted{insertion_mask, w_init_cols(3)}, '.-'); hold on;
    xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold on;
    legend(strrep(w_init_cols, 'w_init_', '\omega_{init,') + "}",'Fontsize',10, 'location', 'bestoutside');
    title(strcat(sprintf("Insertion #%d | ", i), '\omega_{init} vs Insertion Depth'));
    xlabel("Insertion Depth");ylabel("\omega_{init} (1/mm)");
    if save_bool
        savefigas(fig_table_winit, strcat(table_fig_save, "_table_winit.png"));
    end
    
    %plot curvatures(x and y) for channels/AA
    figure(fig_table_curv);
    hold off;
    for aa_k = 1:num_aas
        subplot(2,2,aa_k)
        curv_cols = strcat("Curv_", ["x", "y"], " AA", string(aa_k));
        hold off;
        plot(final_table_sorted{insertion_mask, 'Insertion Depth'},...
             final_table_sorted{insertion_mask, curv_cols(1)}, '.-'); hold on;
        plot(final_table_sorted{insertion_mask, 'Insertion Depth'},...
             final_table_sorted{insertion_mask, curv_cols(2)}, '.-'); hold on;
        xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary'); hold on;
        title(sprintf("AA%d",aa_k));
        xlabel("Insertion Depth");ylabel("Curvature (1/mm)");
        if aa_k == num_aas
            l = legend("x","y",'Fontsize',10, 'orientation', 'horizontal');
            l.Position(1:2) = [(1 - l.Position(3))/2, 0.02];
        end
    end
    sgtitle(sprintf("Insertion #%d | Curvatures vs Insertion Depth", i));
    if save_bool
        savefigas(fig_table_curv, strcat(table_fig_save, "_table_curv.png"));
    end
    
end 
hold off;

%% Summary errorbar plots
% set-up figures
fig_kc_all = figure(14);
set(fig_kc_all, 'units', 'normalized', 'position', [0.25, 0.25, 0.5, 0.5]);
fig_w_init_all = figure(15);
set(fig_w_init_all, 'units', 'normalized', 'position', [0.25 + 1*0.05, 0.25, 0.5, 0.5]);
fig_kc_all_win = figure(16);
set(fig_kc_all_win, 'units', 'normalized', 'position', [0.25 + 2*0.05, 0.25, 0.5, 0.5]);
fig_w_init_all_win = figure(17);
set(fig_w_init_all_win, 'units', 'normalized', 'position', [0.25 + 3*0.05, 0.25, 0.5, 0.5]);

% summarize kappa_c
final_table_sorted_summ = groupsummary(final_table_sorted, {'Insertion Depth', 'Num Layers'}, {'mean', 'std'},...
    {'kc1', 'kc2', 'w_init_1', 'w_init_2', 'w_init_3'});

% masks
depth_mask = final_table_sorted_summ.('Insertion Depth') > 30;
layer2_mask = final_table_sorted_summ.('Num Layers') > 1;

% plot all kappa_c
figure(fig_kc_all);
hold off;
errorbar(final_table_sorted_summ{:, 'Insertion Depth'}, ...
         final_table_sorted_summ.mean_kc1(:), ...
         final_table_sorted_summ.std_kc1(:),...
         'DisplayName', '\kappa_{c,1}'); hold on;
errorbar(final_table_sorted_summ{layer2_mask, 'Insertion Depth'}, ...
         final_table_sorted_summ.mean_kc2(layer2_mask), ...
         final_table_sorted_summ.std_kc2(layer2_mask),...
         'DisplayName', '\kappa_{c,2}'); hold on;
xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary');
legend('location', 'northeastoutside')
xlabel('Insertion Depth (mm)'); ylabel('\kappa_c (1/mm)');
title(sprintf('%s | Double-Layer: \\kappa_c vs. Insertion Depth', strrep(expmt_subdir, '_', ' ')));

% plot all kappa_c windowed
figure(fig_kc_all_win);
hold off;
errorbar(final_table_sorted_summ{depth_mask, 'Insertion Depth'}, ...
         final_table_sorted_summ{depth_mask, 'mean_kc1'}, ...
         final_table_sorted_summ{depth_mask, 'std_kc1'},...
         'DisplayName', '\kappa_{c,1}'); hold on;
errorbar(final_table_sorted_summ{depth_mask & layer2_mask, 'Insertion Depth'}, ...
         final_table_sorted_summ{depth_mask & layer2_mask, 'mean_kc2'}, ...
         final_table_sorted_summ{depth_mask & layer2_mask, 'std_kc2'},...
         'DisplayName', '\kappa_{c,2}'); hold on;
xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary');
legend('Location', 'northeastoutside');
xlabel('Insertion Depth (mm)'); ylabel('\kappa_c (1/mm)');
title(sprintf('%s | Double-Layer: Windowed \\kappa_c vs. Insertion Depth', strrep(expmt_subdir, '_', ' ')));

% plot all w_init
figure(fig_w_init_all);
hold off;
errorbar(final_table_sorted_summ{:,'Insertion Depth'},...
         final_table_sorted_summ.mean_w_init_1, ...
         final_table_sorted_summ.mean_w_init_1, ...
         'DisplayName', '\omega_{init,1}'); hold on;
errorbar(final_table_sorted_summ{:,'Insertion Depth'},...
         final_table_sorted_summ.mean_w_init_2, ...
         final_table_sorted_summ.mean_w_init_2, ...
         'DisplayName', '\omega_{init,2}'); hold on;
errorbar(final_table_sorted_summ{:,'Insertion Depth'},...
         final_table_sorted_summ.mean_w_init_3, ...
         final_table_sorted_summ.mean_w_init_3, ...
         'DisplayName', '\omega_{init,3}'); hold on;
xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary');
legend('Location', 'northeastoutside');
xlabel('Insertion Depth (mm)'); ylabel('\omega_{init} (1/mm)');
title(sprintf('%s | Double-Layer: \\omega_{init} vs. Insertion Depth', strrep(expmt_subdir, '_', ' ')));

% plot all w_init windowed
figure(fig_w_init_all_win);
hold off;
errorbar(final_table_sorted_summ{depth_mask,'Insertion Depth'},...
         final_table_sorted_summ.mean_w_init_1(depth_mask), ...
         final_table_sorted_summ.mean_w_init_1(depth_mask), ...
         'DisplayName', '\omega_{init,1}'); hold on;
errorbar(final_table_sorted_summ{depth_mask,'Insertion Depth'},...
         final_table_sorted_summ.mean_w_init_2(depth_mask), ...
         final_table_sorted_summ.mean_w_init_2(depth_mask), ...
         'DisplayName', '\omega_{init,2}'); hold on;
errorbar(final_table_sorted_summ{depth_mask,'Insertion Depth'},...
         final_table_sorted_summ.mean_w_init_3(depth_mask), ...
         final_table_sorted_summ.mean_w_init_3(depth_mask), ...
         'DisplayName', '\omega_{init,3}'); hold on;
xline(z_crit, 'r--', 'Tissue Boundary', 'DisplayName', 'Tissue Boundary');
legend('Location', 'northeastoutside');
xlabel('Insertion Depth (mm)'); ylabel('\omega_{init} (1/mm)');
title(sprintf('%s | Double-Layer: Windowed \\omega_{init} vs. Insertion Depth', strrep(expmt_subdir, '_', ' ')));

% saving
if save_bool
    fig_cum_fileout_base = fullfile(expmt_dir, strcat(fbgout_basefile, '_cumulative'));
    
    savefigas(fig_kc_all, strcat(fig_cum_fileout_base, '_kc'), ...
              'Verbose', true);
          
    savefigas(fig_kc_all_win, strcat(fig_cum_fileout_base, '_kc-windowed'), ...
              'Verbose', true);
          
    savefigas(fig_w_init_all, strcat(fig_cum_fileout_base, '_winit'), ...
              'Verbose', true);
          
    savefigas(fig_w_init_all_win, strcat(fig_cum_fileout_base, '_winit-windowed'), ...
              'Verbose', true);
end

%% Completion
close all;
disp("Program Terminated.");

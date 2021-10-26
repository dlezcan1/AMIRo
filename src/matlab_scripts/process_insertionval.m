%% process_insertionval.m
%
% this is a script to run through the data points and generate the FBG shape
% from measurements
% 
% - written by: Dimitri Lezcano
configure_env on;
clear;
%% Set-up
% directories to iterate through
expmt_dir = "../../data/3CH-4AA-0004/2021-10-06_Insertion-Expmt-1/"; % CAN CHANGE
trial_dirs = dir(expmt_dir + "Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

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

% calibration matrices matrices file: CAN CHANGE PER NEEDLE
needle_dir = "../../data/3CH-4AA-0004/"; % the needle you are using
needle_calib_file = fullfile(needle_dir, "needle_params_2021-08-16_Jig-Calibration_best.json");

% Initial guesses for kc and w_init DON'T CHANGE
kc_i = 0.002;
w_init_i = [kc_i; 0; 0]; % ideal insertion
theta0 = 0;

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

%% Iterate through the files
lshift = 0;
f3d = figure(1);
set(f3d,'units','normalized','position', [lshift + 0, 0.5, 1/3, .42]);

f2d = figure(2);
set(f2d,'units','normalized','position', [lshift + 1/3, 0.5, 1/3, .42] );

f3d_insert = figure(3);
set(f3d_insert, 'units', 'normalized', 'position', [lshift + 0, 0, 1/3, 0.42]);

f2d_insert = figure(4);
set(f2d_insert, 'units', 'normalized', 'position', [lshift + 2/3, 0, 1/3, 0.42]); 

fkc = figure(5);
set(fkc, 'units', 'normalized', 'position', [lshift + 1/3, 0, 1/3, 0.42]);

fwl_dist = figure(6);
set(fwl_dist, 'units', 'normalized', 'position', [lshift + 2/3, 0.05,1,0.875]);

fwl_shifts = figure(7);
set(fwl_shifts, 'units', 'normalized', 'position', [1, 0.05,1,0.875]);

fcurv = figure(8);
set(fcurv, 'units', 'normalized', 'position', [1, 0.05,1,0.875]);

%%%%%%%%%%%%%%%%%%%%%ALEX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dir_prev = "";
kc_vals = [];
depths = [];
wl_shift_all = []; 
curvatures_all = [];


%%%%%%%%%%%%%%%%%%%ALEX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kc_w_init_final_type = [repmat("double", 1, 38)];
kc_w_init_final_name = ["Insertion Hole","Insertion Depth","kc",...
    "w_init_1","w_init_2","w_init_3",...
    "CH1|AA1","CH1|AA2","CH1|AA3","CH1|AA4",...
    "CH2|AA1","CH2|AA2","CH2|AA3","CH2|AA4",...
    "CH3|AA1","CH3|AA2","CH3|AA3","CH3|AA4",...
    "Curv_x AA1","Curv_y AA1","Curv_x AA2","Curv_y AA2",...
    "Curv_x AA3","Curv_y AA3","Curv_x AA4","Curv_y AA4",...
    "FBG CH1|AA1","FBG CH1|AA2","FBG CH1|AA3","FBG CH1|AA4",...
    "FBG CH2|AA1","FBG CH2|AA2","FBG CH2|AA3","FBG CH2|AA4",...
    "FBG CH3|AA1","FBG CH3|AA2","FBG CH3|AA3","FBG CH3|AA4"];

kc_w_init_final_tbl = table('size',[0,38],'VariableTypes',kc_w_init_final_type...
    ,'VariableNames',kc_w_init_final_name);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(trial_dirs)
    if (~strcmp(dir_prev, trial_dirs(i).folder) && ~strcmp(dir_prev, "")) 
        % plot depths vs kc
        figure(fkc);
        plot(depths, kc_vals, '*-');
        xlabel("Insertion Depth (mm)"); ylabel("\kappa_c (1/mm)");
        title(sprintf("Insertion #%d", hole_num) + " | \kappa_c vs Insertion Depth");
        
        % plot the wavelength and curvatures shifts
        for aa_i = 1:num_aas
            %ch_idxs = aa_i:(num_chs):(num_chs*num_aas);
            ch_idxs = [1,4,7,10];
            
            figure(fwl_shifts);
            subplot(1,num_aas,aa_i);
            
            %%%%%%%%%%%ALEX%%%%%%%%%%%%
            %wl_shift_all_plot = reshape(wl_shift_all(:,ch_idxs),1,[]);
            %plot(reshape(depths,1,[]), wl_shift_all_plot,'*-');
            %xlabel("Insertion Depth (mm)"); ylabel("Wavelength Shift T-Comp. (nm)");
            %title(sprintf("AA%d", aa_i), 'FontSize', 20);
            
        end
        CH_labels = strcat("CH", string(1:num_chs));
        legend(CH_labels, 'Location', 'bestoutside');
        
        figure(fcurv);
        subplot(1,2,1);
        plot(depths, squeeze(curvatures_all(1,:,:)),'*-');
        title('Curvature X (1/m)');
        xlabel('Insertion Depth (mm)'); ylabel("Curvature (1/m)");
        
        subplot(1,2,2);
        plot(depths, squeeze(curvatures_all(2,:,:)),'*-');
        legend(CH_labels, 'Location', 'bestoutside');
        title('Curvature Y (1/m)');
        xlabel('Insertion Depth (mm)'); 
        
        % save the figures
        if save_bool
           savefigas(fkc, strcat(fileout_base, "_kc-all-insertions.png"));
           
        end
        
        % empty the values
        %kc_vals = [];
        %depths = [];
        kc_vals = [];
        depths = [];
        wl_shift_all = []; 
        curvatures_all = [];
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
       T = table(kc, w_init', theta0, L, 'VariableNames', ...
           {'kc', 'w_init', 'theta0', 'L'});
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
    
    if i == length(trial_dirs) % handle the last edge case
        % plot depths vs kc
        figure(fkc);
        plot(depths, kc_vals, '*-');
        xlabel("Insertion Depth (mm)"); ylabel("\kappa_c (1/mm)");
        title(sprintf("Insertion #%d", hole_num) + " | \kappa_c vs Insertion Depth");
        
        % plot the wavelength and curvatures shifts
        for aa_i = 1:num_aas
            ch_idxs = aa_i:num_chs:num_chs*num_aas;
            
            %figure(fwl_shifts);
            %subplot(1,num_aas,aa_i);
            %plot(depths, wl_shift_all(:,ch_idxs),'*-');
            %xlabel("Insertion Depth (mm)"); ylabel("Wavelength Shift T-Comp. (nm)");
            %title(sprintf("AA%d", aa_i), 'FontSize', 20);
            
        end
        CH_labels = strcat("CH", string(1:num_chs));
        legend(CH_labels, 'Location', 'bestoutside');
        
        figure(fcurv);
        subplot(1,2,1);
        plot(depths, squeeze(curvatures_all(1,:,:)),'*-');
        title('Curvature X (1/m)');
        xlabel('Insertion Depth (mm)'); ylabel("Curvature (1/m)");
        
        subplot(1,2,2);
        plot(depths, squeeze(curvatures_all(2,:,:)),'*-');
        legend(CH_labels, 'Location', 'bestoutside');
        title('Curvature Y (1/m)');
        xlabel('Insertion Depth (mm)'); 
        
        % save the figures
        if save_bool
           savefigas(fkc, strcat(fileout_base, "_kc-all-insertions.png"));
           
        end
        
        % empty the values
        kc_vals = [];
        depths = [];
        kc_vals = [];
        depths = [];
        wl_shift_all = []; 
        curvatures_all = [];
    end
    %%%%%%%%%%%%%%%ALEX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fbg_data = cell(1,12);
    for i = 1:12
        fbg_data{i} = wls_mat(:,i);
    end
    
    %appending result to kc/w_init table
    kc_w_init_final_tbl = [kc_w_init_final_tbl; 
        {hole_num,ins_depth,kc,w_init(1),w_init(2),w_init(3),...
        wl_shift_Tcorr(1,1),wl_shift_Tcorr(1,2),wl_shift_Tcorr(1,3),wl_shift_Tcorr(1,4),...
        wl_shift_Tcorr(2,1),wl_shift_Tcorr(2,2),wl_shift_Tcorr(2,3),wl_shift_Tcorr(2,4),...
        wl_shift_Tcorr(3,1),wl_shift_Tcorr(3,2),wl_shift_Tcorr(3,3),wl_shift_Tcorr(3,4),...
        curvatures(1,1),curvatures(2,1),curvatures(1,2),curvatures(2,2),...
        curvatures(1,3),curvatures(2,3),curvatures(1,4),curvatures(2,4),...
        fbg_data{1},fbg_data{2},fbg_data{3},fbg_data{4},...
        fbg_data{5},fbg_data{6},fbg_data{7},fbg_data{8},...
        fbg_data{9},fbg_data{10},fbg_data{11},fbg_data{12}}];
    
  
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % add to kappa c records
    kc_vals = [kc_vals; kc];
    depths = [depths; ins_depth];
    wl_shift_all = cat(3,wl_shift_all, wl_shift_Tcorr);
    curvatures_all = cat(3, curvatures_all, curvatures);
    
    % output
    fprintf("Finished trial: '%s' in %.2f secs.\n", d, t);
    disp(" ");
    
    
end
%% Plotting from table
final_table = table2array(kc_w_init_final_tbl(:,1:26));
final_table_sorted = sortrows(final_table,[1,2]);

fig_table_wls = figure(9);
fig_table_kc = figure(10);
fig_table_winit = figure(11);
fig_table_curv = figure(12);

for i = 1:9
    ii = i-1;Nprev = 0;
    index = int2str(i);
    table_fig_save = fullfile(expmt_dir, sprintf("Insertion%d/", i));
    insertion_test = (final_table_sorted(:,1)== i);
    N = length(insertion_test(insertion_test == 1));
    
    % plot wls_shift_tcorr for channels/AA
    figure(fig_table_wls)
    hold off;
    for k = 1:4
        subplot(2,2,k)
        hold off;
        plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,6+k),'.-'); hold on;
        plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,10+k),'.-'); hold on;
        plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,14+k),'.-'); hold on;
        title(sprintf("AA%d",k));
        xlabel("Insertion Depth");ylabel("Wavelength Shift");
        legend("CH1","CH2","CH3",'Fontsize',5);
    end
    sgtitle(sprintf("Insertion #%d | Wavelength shift", i));
    savefigas(fig_table_wls, fullfile(table_fig_save, "_table_wls_shift"));
        
    % plot kc
    figure(fig_table_kc);
    hold off;
    plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,3),'*-');
    title(sprintf("Insertion #%d | \kappa_c vs Insertion Depth", i));
    xlabel("Insertion Depth");ylabel("\kappa_c (1/mm)");
    savefigas(fig_table_kc, strcat(table_fig_save, "_table_kc.png"));
    
    % plot w_init1,2,3
    figure(fig_table_winit);
    hold off;
    plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,4),'.-'); hold on;
    plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,5),'.-'); hold on;
    plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,6),'.-'); hold on;
    legend("w_init1","w_init2","w_init3",'Fontsize',5);
    
    title(strcat(sprintf("Insertion #%d | ", i), '\omega_{init} vs Insertion Depth'));
    xlabel("Insertion Depth");ylabel("\omega_{init} (1/mm)");
    savefigas(fig_table_winit, strcat(table_fig_save, "_table_winit.png"));
    
    %plot curvatures(x and y) for channels/AA
    figure(fig_table_curv);
    hold off;
    for k = 1:4
        subplot(2,2,k)
        hold off;
        plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,17+2*k),'.-'); hold on;
        plot(final_table_sorted(1+ii*Nprev:N+ii*Nprev,2),final_table_sorted(1+ii*Nprev:N+ii*Nprev,18+2*k),'.-'); hold on;
        title(sprintf("AA%d",k));
        legend("x","y",'Fontsize',5);
        xlabel("Insertion Depth");ylabel("Curvature (1/mm)");
    end
    sgtitle(sprintf("Insertion #%d | Curvatures vs Insertion Depth", i));
    savefigas(fig_table_curv, strcat(table_fig_save, "_table_curv.png"));
    
    i = i+1;
    Nprev = N;
end 
hold off;
%close all;
%% Completion
close all;
disp("Program Terminated.");

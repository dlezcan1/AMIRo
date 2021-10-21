%% summarize_singlelayer_insertion.m
% - written by: Dimitri Lezcano
%
% summarize the insertion data for single-layer single-bend insertions
%
% 

%% Set-up
% experiment directories
expmt_dir = fullfile("../../data/3CH-4AA-0004/2021-10-08_Insertion-Expmt-1");
split_expmt_dir = regexp(expmt_dir, filesep, 'split');

% options
save_bool = true;

% filenames
result_tbl_file = fullfile(expmt_dir, "kc_w_init_final_table.mat");
baseout_name = fullfile(expmt_dir, "experiment_results_single-layer");

% Plot configurations for experiment (for cases
depth_xl = 65; % yline depth
xl_desc = "Rotate 180^o";
% xl_desc = "Tissue Boundary";

% Insertion depth windows (for windows plotting)
depth_lb = 30;
depth_ub = inf;

%% Load the results
kc_w_init_final_tbl = load(result_tbl_file, 'kc_w_init_final_tbl').kc_w_init_final_tbl;

%% Summarize the results
kc_summ_tbl = groupsummary(kc_w_init_final_tbl, 'Insertion Depth', {'mean', 'std'},...
                            {'kc', 'w_init_1', 'w_init_2', 'w_init_3'});

%% Plotting
% figure configuration
screensize = get(0, 'Screensize');
fig_size = [600 , 485];
fig_l_step = fig_size(1);
fig_b_step = -fig_size(2) + 50;
l_offset = 0;
b_offset = 975 - fig_size(2);
n_figcols = round(screensize(3)/fig_size(1));

% Plot the mean kc values
fig_counter = 1;
fig_kc = figure(fig_counter);
col_idx = mod(fig_counter - 1, n_figcols);
row_idx = max(fig_counter - 1 - n_figcols*col_idx, 0);
set(fig_kc, 'Position', [l_offset + fig_l_step * col_idx, ...
                         b_offset + fig_b_step * row_idx,...
                         fig_size(1), fig_size(2)]);
fig_counter = fig_counter + 1;

errorbar(kc_summ_tbl.("Insertion Depth"), kc_summ_tbl.mean_kc, kc_summ_tbl.std_kc);
if depth_xl > min(kc_summ_tbl.("Insertion Depth"))
   plot_xline(depth_xl, xl_desc);
end

xlabel('Insertion Depth (mm)'); ylabel('\kappa_c (1/mm)');
title(strcat('\kappa_c ', strrep(sprintf('for Experiment: %s', split_expmt_dir(end)), '_', ' ')));

% windowed mean kc values
fig_kc_win = figure(fig_counter);
col_idx = mod(fig_counter - 1, n_figcols);
row_idx = max(fig_counter - 1 - n_figcols*col_idx, 0);
set(fig_kc_win, 'Position', [l_offset + fig_l_step * col_idx, ...
                         b_offset + fig_b_step * row_idx,...
                         fig_size(1), fig_size(2)]);
fig_counter = fig_counter + 1;

errorbar(kc_summ_tbl.("Insertion Depth"), kc_summ_tbl.mean_kc, kc_summ_tbl.std_kc);
if depth_xl > min(kc_summ_tbl.("Insertion Depth"))
   plot_xline(depth_xl, xl_desc);
end

xlabel('Insertion Depth (mm)'); ylabel('\kappa_c (1/mm)');
xlim([depth_lb, depth_ub]);
title(strcat('Windowed \kappa_c ', strrep(sprintf('for Experiment: %s', split_expmt_dir(end)), '_', ' ')));

% Plot the mean w_init values
fig_winit = figure(fig_counter);
col_idx = mod(fig_counter - 1, n_figcols);
row_idx = max(fig_counter - 1 - n_figcols*col_idx, 0);
set(fig_winit, 'Position', [l_offset + fig_l_step * col_idx, ...
                         b_offset + fig_b_step * row_idx,...
                         fig_size(1), fig_size(2)]);
fig_counter = fig_counter + 1;
hold off;
for i = 1:3
    errorbar(kc_summ_tbl.("Insertion Depth"), ...
             kc_summ_tbl.(sprintf('mean_w_init_%d', i)), ...
             kc_summ_tbl.(sprintf('std_w_init_%d', i))); hold on;
end

if depth_xl > min(kc_summ_tbl.("Insertion Depth"))
   plot_xline(depth_xl, xl_desc);
end

xlabel('Insertion Depth (mm)'); ylabel('\omega_{init} (1/mm)');
title(strcat('\omega_{init} ', strrep(sprintf('for Experiment: %s', split_expmt_dir(end)), '_', ' ')));
w_init_lbls = strcat("\omega_{init,", string(1:3)', '}');
legend(w_init_lbls);

% Plot the mean w_init values
fig_winit_win = figure(fig_counter);
col_idx = mod(fig_counter - 1, n_figcols);
row_idx = floor((fig_counter -1)/n_figcols);
set(fig_winit_win, 'Position', [l_offset + fig_l_step * col_idx, ...
                         b_offset + fig_b_step * row_idx,...
                         fig_size(1), fig_size(2)]);
fig_counter = fig_counter + 1;
hold off;
for i = 1:3
    errorbar(kc_summ_tbl.("Insertion Depth"), ...
             kc_summ_tbl.(sprintf('mean_w_init_%d', i)), ...
             kc_summ_tbl.(sprintf('std_w_init_%d', i))); hold on;
end

if depth_xl > min(kc_summ_tbl.("Insertion Depth"))
   plot_xline(depth_xl, xl_desc);
end

xlabel('Insertion Depth (mm)'); ylabel('\omega_{init} (1/mm)');
xlim([depth_lb, depth_ub]);
title(strcat('\omega_{init} ', strrep(sprintf('for Experiment: %s', split_expmt_dir(end)), '_', ' ')));
w_init_lbls = strcat("\omega_{init,", string(1:3)', '}');
legend(w_init_lbls);

%% Saving
if save_bool
    savefigas(fig_kc, strcat(baseout_name, '_kc'), 'Verbose', true);
    
    savefigas(fig_kc_win, strcat(baseout_name, '_kc-windowed'), 'Verbose', true);
    
    savefigas(fig_winit, strcat(baseout_name, '_winit'), 'Verbose', true);
    
    savefigas(fig_winit_win, strcat(baseout_name, '_winit-windowed'), 'Verbose', true);

end

%% Helper function
function plot_xline(xl, desc)
    hold on; xline(xl, 'r--', desc); hold off;
end
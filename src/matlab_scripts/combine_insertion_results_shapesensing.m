%% combine_insertion_results_shapesensing.m
%
% This is a script to combine the data tables from the results of insertion
%   experiments
%
% - written by: Dimitri Lezcano

clear; 
%% Set-up
needle_dir = fullfile('../../data', '3CH-4AA-0004'); % CHANGE ONLY ARG 2
expmt_dirs = dir(fullfile(needle_dir, '*_Insertion-Expmt-*')); % CHANGE ONLY ARG 2

% Insertion directory pattern
exclude_dirs = {'2021-08-24_Insertion-Expmt-1','2021-08-25_Insertion-Expmt-1',...
                '2021-08-25_Insertion-Expmt-2','2021-08-30_Insertion-Expmt-1',...
                '2021-10-01_Insertion-Expmt-1', '2021-10-13_Insertion-Expmt-1',...
                '',...
                }; % CAN CHANGE
mask_exclude = any(strcmp(repmat(exclude_dirs', 1, numel(expmt_dirs)), ...
                          repmat({expmt_dirs.name}, numel(exclude_dirs), 1)), 1);
expmt_dirs = expmt_dirs(~mask_exclude); 

% parameters CAN CHANGE
use_weights = true; % use AA reliability weightngs
save_bool = true; % whether to save the data
save_dir = fullfile(needle_dir, 'Insertion_Experiment_Results'); % DON'T CHANGE
if save_bool && ~isfolder(save_dir)
    mkdir(save_dir);
end

% base file names
experiment_file = 'experiment.json';
dataout_file = "FBGdata_shapesensing";
if use_weights
    result_file_base = '%s_%s_FBG-Camera-Comp_tip-pcr_FBG-weights_results.mat';
    insertion_file_base = 'FBGdata%s_FBG-weights_results.mat';
else
    result_file_base = '%s_%s_FBG-Camera-Comp_tip-pcr_results.mat';
    insertion_file_base = 'FBGdata%s_results.mat';
end

result_excel_file_base = strrep(result_file_base, '.mat', '.xlsx');

combined_result_file = 'FBG-Camera-Comp_tip-pcr_FBG-weights_combined-results.mat';

% recalculate
recalculate = true; % CAN CHANGE

%% Prepare data table entries
% - actual result table
act_col_names = {'TissueStiffness1', 'TissueStiffness2', 'Experiment', 'Ins_Hole', 'L_ref', ...
                 'w_init_1', 'w_init_2', 'w_init_3', 'kc1', 'kc2', ...
                 'num_layers', 'singlebend',...
                 'cam_shape', 'fbg_shape', 'Pose_nc',...
                 'RMSE', 'MaxError', 'InPlane', 'OutPlane'};
Nact_cols = numel(act_col_names);
Nact_err  = Nact_cols - find(strcmp(act_col_names, 'RMSE')) + 1;
act_col_err_names = act_col_names(end-Nact_err+1:end);
actcol_types = cat(2, {'double','double', 'string', 'uint8',...
                'double','double','double', ...
                'double','double','double','uint8','logical',...
                'double','double','double'}, repmat({'double'}, 1, Nact_err));
actcol_units = cat(2, {'units', 'units', '', '','mm',...
                       '1/mm','1/mm','1/mm','1/mm','1/mm',...
                       '','','mm','mm','mm'}, repmat({'mm'}, 1, Nact_err));
     
%% Load the current data matrix
if isfile(fullfile(needle_dir, 'Insertion_Experiment_Results', combined_result_file)) && ~recalculate
    l = load(fullfile(needle_dir, 'Insertion_Experiment_Results', combined_result_file));
    
    act_result_tbl = l.act_result_tbl;
    
    clear l;
    
else
    % empty tables
    act_result_tbl = table('size', [0, Nact_cols], 'VariableTypes', actcol_types,...
                           'VariableNames', act_col_names);
    act_result_tbl.Properties.VariableUnits = actcol_units;
    act_result_tbl.Properties.Description = ...
        "Comparison between Actual FBG shapes and stereo reconstruction";
    
end

%% Iterate through the Experiments
for i = 1:numel(expmt_dirs)
    fprintf("Processing: %s... ", expmt_dirs(i).name);

    % load in the experiment description
    expmt_desc = jsondecode(fileread(fullfile(expmt_dirs(i).folder, ...
                                              expmt_dirs(i).name,...
                                              experiment_file)));
    
    % determine type of insertion
    if isfield(expmt_desc, 'DoubleBendDepth') % double-bend
        singlebend = false;
        layers = 1;
        stiffness1 = expmt_desc.OODurometer;
        stiffness2 = -1;
        result_file = sprintf(result_file_base, 'DoubleBend', 'SingleLayer');
        insertion_file = sprintf(insertion_file_base,'_doublebend');
    
    elseif isfield(expmt_desc, 'tissue1Length') % single-bend double-layer
        singlebend = true;
        layers = 2;
        stiffness1 = expmt_desc.OODurometer1;
        stiffness2 = expmt_desc.OODurometer2;
        result_file = sprintf(result_file_base, 'SingleBend', 'DoubleLayer');
        insertion_file = sprintf(insertion_file_base,'_2layer');
        
    else % single-bend single-layer 
        singlebend = true;
        layers = 1;
        stiffness1 = expmt_desc.OODurometer;
        stiffness2 = -1;
        result_file = sprintf(result_file_base, 'SingleBend', 'SingleLayer');
        insertion_file = sprintf(insertion_file_base,'_1layer');
        
    end
    
    % load experiment results
    expmt_results = load(fullfile(expmt_dirs(i).folder, expmt_dirs(i).name, ...
                        result_file));
    expmt_results.insertion_tbl = load(fullfile(expmt_dirs(i).folder, expmt_dirs(i).name,...
                        insertion_file)).final_table_sorted;
    if singlebend
        expmt_results.act_result_tbl = join(expmt_results.fbg_cam_compare_tbl,...
                                            expmt_results.insertion_tbl,...
                                            'LeftKeys', {'Ins_Hole', 'L'},...
                                            'RightKeys', {'Insertion Hole', 'Insertion Depth'});
    else % doublebend
        expmt_results.insertion_tbl.singlebend = ~expmt_results.insertion_tbl.DoubleBend;
        expmt_results.act_result_tbl = join(expmt_results.fbg_cam_compare_tbl,...
                                            expmt_results.insertion_tbl,...
                                            'LeftKeys', {'Ins_Hole', 'L', 'singlebend'},...
                                            'RightKeys', {'Insertion Hole', 'Insertion Depth', 'singlebend'});
    end
                                    
    % format the new table
    old_name = {'L','MeanInPlane','MeanOutPlane'}; 
    new_name = {'L_ref','InPlane','OutPlane'};
    if any(strcmp(expmt_results.act_result_tbl.Properties.VariableNames, 'kc'))
        old_name{end+1} = 'kc'; new_name{end+1} = 'kc1';
    end
    expmt_results.act_result_tbl_fmt = renamevars(expmt_results.act_result_tbl,...
                                                  old_name, new_name);
    expmt_results.act_result_tbl_fmt.TissueStiffness1(:) = stiffness1;
    expmt_results.act_result_tbl_fmt.TissueStiffness2(:) = stiffness2;
    expmt_results.act_result_tbl_fmt.num_layers(:) = layers;
    expmt_results.act_result_tbl_fmt.Experiment(:) = string(expmt_dirs(i).name);
    
    if ~any(strcmp(expmt_results.act_result_tbl_fmt.Properties.VariableNames, 'singlebend'))
        expmt_results.act_result_tbl_fmt.singlebend(:) = singlebend;
    end
    
    if layers < 2
        expmt_results.act_result_tbl_fmt.kc2(:) = -1;
    end
        
    
    % rename the columns and keep relevant columns only
    expmt_results.act_result_tbl_fmt_rel = expmt_results.act_result_tbl_fmt(:,act_col_names);
        
    % add the columns to current experimental results table
    Nact_rows = size(expmt_results.act_result_tbl_fmt_rel,1);
            
    % remove any current entries for this experiment
    if any(strcmp(act_result_tbl.Experiment, expmt_dirs(i).name)) 
        if recalculate
            act_result_tbl(strcmp(act_result_tbl.Experiment, expmt_dirs(i).name),:) = [];
        else
            disp("Skipping.");
            disp(" ");
            continue;
        end
    else
        % add to the current table
        act_result_tbl = [act_result_tbl; expmt_results.act_result_tbl_fmt_rel];


        disp("Completed.");
        disp(" ");
    end
    

end
% return;

%% Perform statistics
% actual results
act_result_summ_Lref = groupsummary(act_result_tbl, 'L_ref', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_InsHole = groupsummary(act_result_tbl, 'Ins_Hole', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_stiff1 = groupsummary(act_result_tbl, 'TissueStiffness1', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_stiff2 = groupsummary(act_result_tbl, 'TissueStiffness2', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_stiff2 = act_result_summ_stiff2(act_result_summ_stiff2.TissueStiffness2 > 0,:);


%% Save the Data
if save_bool
    % .mat file
    save(fullfile(save_dir, combined_result_file), 'act_result_tbl');
    fprintf("Saved data to: %s\n", fullfile(save_dir, combined_result_file));
    
%     % tables
%     writetable(act_result_tbl, fullfile(save_dir, result_excel_file_base),...
%         'WriteVariableNames',true, 'Sheet', 'Act. All Results');
%     
%     writetable(pred_result_tbl, fullfile(save_dir, result_excel_file_base),...
%         'WriteVariableNames',true, 'Sheet', 'Pred. All Results');
%     fprintf("Wrote data to: %s\n", fullfile(save_dir, result_excel_file_base));
    
end

%% Plot the Statistics (Actual Results)
fig_counter = 1;
ax_pos_adj = [0, 0, 0, -0.05]; % adjust the height of the position
max_yl = 1.25;

% Actual Errors per Insertion Hole
fig_act_inshole = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_act_inshole, 'units', 'normalized', 'position',[ -1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_act_inshole_rmse = subplot(1,3,1);
ax_act_inshole_rmse.Position = ax_act_inshole_rmse.Position + ax_pos_adj;
boxplot(act_result_tbl.RMSE, act_result_tbl.Ins_Hole);
xlabel("Insertion Hole Number"); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% - In-Plane Error
ax_act_inshole_inplane = subplot(1,3,2);
ax_act_inshole_inplane.Position = ax_act_inshole_inplane.Position + ax_pos_adj;
boxplot(act_result_tbl.InPlane, act_result_tbl.Ins_Hole);
xlabel("Insertion Hole Number"); ylabel("In-Plane Error(mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% - Out-of-Plane Error
ax_act_inshole_outplane = subplot(1,3,3);
ax_act_inshole_outplane.Position = ax_act_inshole_outplane.Position + ax_pos_adj;
boxplot(act_result_tbl.OutPlane, act_result_tbl.Ins_Hole);
xlabel("Insertion Hole Number"); ylabel("Out-of-Plane Error(mm)");
title("Mean Out-of-Plane Errors", 'fontsize', 20);

% - titling and limits
sgtitle("Errors between Stereo and FBG Reconstructed Shapes per Insertion Hole",...
    'FontSize', 22, 'FontWeight', 'bold')
ax_act_inshole_rmse.YLim = [0, max([ax_act_inshole_rmse.YLim,...
                                    ax_act_inshole_inplane.YLim,...
                                    ax_act_inshole_outplane.YLim,...
                                    max_yl])];
ax_act_inshole_inplane.YLim = ax_act_inshole_rmse.YLim;
ax_act_inshole_outplane.YLim = ax_act_inshole_rmse.YLim;


% Actual Errors per Insertion Depths
fig_act_Lref = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_act_Lref, 'units', 'normalized', 'position',[ -1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_act_Lref_rmse = subplot(1,3,1);
ax_act_Lref_rmse.Position = ax_act_Lref_rmse.Position + ax_pos_adj;
boxplot(act_result_tbl.RMSE, act_result_tbl.L_ref);
xlabel("Insertion Depth (mm)"); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% - In-Plane Error
ax_act_Lref_inplane = subplot(1,3,2);
ax_act_Lref_inplane.Position = ax_act_Lref_inplane.Position + ax_pos_adj;
boxplot(act_result_tbl.InPlane, act_result_tbl.L_ref);
xlabel("Insertion Depth (mm)"); ylabel("In-Plane Error(mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% - Out-of-Plane Error
ax_act_Lref_outplane = subplot(1,3,3);
ax_act_Lref_outplane.Position = ax_act_Lref_outplane.Position + ax_pos_adj;
boxplot(act_result_tbl.OutPlane, act_result_tbl.L_ref);
xlabel("Insertion Depth (mm)"); ylabel("Out-of-Plane Error(mm)");
title("Mean Out-of-Plane Errors", 'fontsize', 20);

% - titling and limits
sgtitle("Errors between Stereo and FBG Reconstructed Shapes per Insertion Depth",...
    'FontSize', 22, 'FontWeight', 'bold')
ax_act_Lref_rmse.YLim = [0, max([ax_act_Lref_rmse.YLim,...
                                    ax_act_Lref_inplane.YLim,...
                                    ax_act_Lref_outplane.YLim,...
                                    max_yl])];
ax_act_Lref_inplane.YLim = ax_act_Lref_rmse.YLim;
ax_act_Lref_outplane.YLim = ax_act_Lref_rmse.YLim;


% Actual Errors per Tissue Stiffness
fig_act_stiff = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_act_stiff, 'units', 'normalized', 'position',[ -1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_act_stiff_rmse = subplot(1,3,1);
ax_act_stiff_rmse.Position = ax_act_stiff_rmse.Position + ax_pos_adj;
boxplot(act_result_tbl.RMSE, [act_result_tbl.TissueStiffness1, act_result_tbl.TissueStiffness2]);
xlabel("Tissue Stiffness (1 & 2) (OO units)"); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% - In-Plane Error
ax_act_stiff_inplane = subplot(1,3,2);
ax_act_stiff_inplane.Position = ax_act_stiff_inplane.Position + ax_pos_adj;
boxplot(act_result_tbl.InPlane, [act_result_tbl.TissueStiffness1, act_result_tbl.TissueStiffness2, act_result_tbl.singlebend]);
xlabel("Tissue Stiffness (1 & 2) (OO units)"); ylabel("In-Plane Error(mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% - Out-of-Plane Error
ax_act_stiff_outplane = subplot(1,3,3);
ax_act_stiff_outplane.Position = ax_act_stiff_outplane.Position + ax_pos_adj;
boxplot(act_result_tbl.OutPlane, [act_result_tbl.TissueStiffness1, act_result_tbl.TissueStiffness2]);
xlabel("Tissue Stiffness (1 & 2) (OO units)"); ylabel("Out-of-Plane Error(mm)");
title("Mean Out-of-Plane Errors", 'fontsize', 20);

% - titling and limits
sgtitle("Errors between Stereo and FBG Reconstructed Shapes per Tissue Stiffness",...
    'FontSize', 22, 'FontWeight', 'bold')
ax_act_stiff_rmse.YLim = [0, max([ax_act_stiff_rmse.YLim,...
                                    ax_act_stiff_inplane.YLim,...
                                    ax_act_stiff_outplane.YLim,...
                                    max_yl])];
ax_act_stiff_inplane.YLim = ax_act_stiff_rmse.YLim;
ax_act_stiff_outplane.YLim = ax_act_stiff_rmse.YLim;

% plotting per Insertion Experiment type
fig_act_exptype = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_act_exptype, 'units', 'normalized', 'position',[ -1, 0.0370, 1.0000, 0.8917 ]);
singlebend_category = categorical(act_result_tbl.singlebend, [0,1], ...
                                  {'S-Shape', 'C-Shape'}, 'ordinal', true);
tissue_type = ordinal([act_result_tbl.TissueStiffness1, act_result_tbl.TissueStiffness2],...
                       {'None', 'Soft', 'Hard'},[], [-inf, 20, 29, inf]);
exp_types = categorical([string(tissue_type), string(singlebend_category)]);
                            

% - RMSE
ax_act_exptype_rmse = subplot(1,3,1);
ax_act_exptype_rmse.Position = ax_act_exptype_rmse.Position + ax_pos_adj;
boxplot(act_result_tbl.RMSE, exp_types);
xlabel("Tissue Hardness (1st, 2nd Layer) and Shape"); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% - In-Plane Error
ax_act_exptype_inplane = subplot(1,3,2);
ax_act_exptype_inplane.Position = ax_act_exptype_inplane.Position + ax_pos_adj;
boxplot(act_result_tbl.InPlane, exp_types);
xlabel("Tissue Hardness (1st, 2nd Layer) and Shape"); ylabel("In-Plane Error(mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% - Out-of-Plane Error
ax_act_exptype_outplane = subplot(1,3,3);
ax_act_exptype_outplane.Position = ax_act_exptype_outplane.Position + ax_pos_adj;
boxplot(act_result_tbl.OutPlane, exp_types);
xlabel("Tissue Hardness (1st, 2nd Layer) and Shape"); ylabel("Out-of-Plane Error(mm)");
title("Mean Out-of-Plane Errors", 'fontsize', 20);

% - titling and limits
sgtitle("Errors between Stereo and FBG Reconstructed Shapes per Tissue Stiffness",...
    'FontSize', 22, 'FontWeight', 'bold')
ax_act_exptype_rmse.YLim = [0, max([ax_act_exptype_rmse.YLim,...
                                    ax_act_exptype_inplane.YLim,...
                                    ax_act_exptype_outplane.YLim,...
                                    max_yl])];
ax_act_exptype_inplane.YLim = ax_act_exptype_rmse.YLim;
ax_act_exptype_outplane.YLim = ax_act_exptype_rmse.YLim;

%% Save the plots
if save_bool
    dataout_dir_file = fullfile(save_dir, dataout_file);
    
    % Actual Results
    savefigas(fig_act_inshole, strcat(dataout_dir_file, '_actual-error_inshole'),...
        'Verbose', true);
        
    savefigas(fig_act_Lref, strcat(dataout_dir_file, '_actual-error_Lref'),...
        'Verbose', true);
        
    savefigas(fig_act_stiff, strcat(dataout_dir_file, '_actual-error_stiff'),...
        'Verbose', true);
    
    savefigas(fig_act_exptype, strcat(dataout_dir_file, '_actual-error_expmt-type'),...
        'Verbose', true);
    
        
end

%% Termination
disp("Press [ENTER] to finish the program");
pause;
close all;
disp("Program Completed.");
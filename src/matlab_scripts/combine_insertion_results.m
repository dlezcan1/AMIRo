
%% combine_insertion_results.m
%
% This is a script to combine the data tables from the results of insertion
%   experiments
%
% - written by: Dimitri Lezcano

clear; 
%% Set-up
needle_dir = fullfile('../../data/', '3CH-4AA-0004'); % CHANGE ONLY ARG 2
expmt_dirs = dir(fullfile(needle_dir, '*_Insertion-Expmt-*')); % CHANGE ONLY ARG 2

% Insertion directory pattern
exclude_dirs = {'2021-08-25_Insertion-Expmt-1', '2021-08-25_Insertion-Expmt-2',...
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
dataout_file = "FBGdata_prediction";
if use_weights
    dataout_file = strcat(dataout_file, "_FBG-weights");
end
result_file = strcat(dataout_file, '_results.mat');
result_excel_file = strrep(result_file, '.mat', '.xlsx');

%% Prepare data table entries
% - actual result table
act_col_names = {'TissueStiffness', 'Experiment', 'Ins_Hole', 'L_ref', ...
         'cam_shape', 'fbg_shape', 'RMSE', 'MaxError', 'InPlane', 'OutPlane'};
Nact_cols = numel(act_col_names);
Nact_err  = Nact_cols - find(strcmp(act_col_names, 'RMSE')) + 1;
act_col_err_names = act_col_names(end-Nact_err+1:end);
actcol_types = cat(2, {'double', 'string', 'uint8'}, repmat({'double'}, 1, Nact_cols-3));
actcol_units = cat(2, {'units', '', ''}, repmat({'mm'}, 1, Nact_cols-3));
     
% - predicted result table
pred_col_names = {'TissueStiffness', 'Experiment', 'Ins_Hole', 'L_ref', 'L_pred', ...
             'cam_shape', 'fbgref_shape', 'pred_shape', 'q_optim', ...
             'FBG_RMSE', 'FBG_MaxError', 'FBG_TipError', 'FBG_InPlaneError', 'FBG_OutPlaneError', ...
             'Cam_RMSE', 'Cam_MaxError', 'Cam_TipError', 'Cam_InPlaneError', 'Cam_OutPlaneError'};
Npred_cols = numel(pred_col_names);
Npred_err = Npred_cols - find(strcmp(pred_col_names, "FBG_RMSE")) + 1 ; % number of error columns
pred_col_err_names = pred_col_names(end-Npred_err+1:end);
predcol_types = cat(2, {'double', 'string', 'uint8'}, repmat({'double'}, 1, Npred_cols-3));
predcol_units = {'units','', '', 'mm', 'mm', 'mm', 'mm', 'mm', ''};
predcol_units = cat(2, predcol_units, repmat({'mm'}, 1, Npred_err));

%% Load the current data matrix
if isfile(fullfile(needle_dir, result_file))
    l = load(fullfile(needle_dir, result_file));
    
    act_result_tbl = l.act_result_tbl;
    pred_result_tbl = l.pred_result_tbl;
    
    clear l;
    
else
    % empty tables
    act_result_tbl = table('size', [0, Nact_cols], 'VariableTypes', actcol_types,...
                           'VariableNames', act_col_names);
    act_result_tbl.Properties.VariableUnits = actcol_units;
    act_results_tbl.Properties.Description = ...
        "Comparison between Actual FBG shapes and stereo reconstruction";
    
    pred_result_tbl = table('Size', [0,Npred_cols], 'VariableTypes', predcol_types, ...
        'VariableNames', pred_col_names);
    pred_result_tbl.Properties.VariableUnits = predcol_units;
    pred_results_tbl.Properties.Description = ...
        "Comparison between Actual FBG shapes, stereo reconstruction, and predicted FBG shapes.";
    
end

%% Iterate through the Experiments
for i = 1:numel(expmt_dirs)
    fprintf("Processing: %s... ", expmt_dirs(i).name);

    % load in the results and experiment
    expmt_results = load(fullfile(expmt_dirs(i).folder, expmt_dirs(i).name, ...
                        result_file));
    expmt_desc = jsondecode(fileread(fullfile(expmt_dirs(i).folder, ...
                                              expmt_dirs(i).name,...
                                              experiment_file)));
                                          
    % determine type of insertion
    if isfield(expmt_desc, 'DoubleBendDepth') % double-bend
        singlebend = false;
        layers = 1;
        stiffness1 = expmt_desc.OODurometer;
    
    elseif isfield(expmt_desc, 'tissue1Length')
        singlebend=true;
        layers=2;
        stiffness1 = expmt_desc.OODur
    
    % get the Tissue Stiffness
    stiffness1 = expmt_desc.OODurometer;
    
    % add the columns to current experimental results table
    Nact_rows = size(expmt_results.act_result_tbl,1);
    Npred_rows = size(expmt_results.pred_result_tbl,1);
    
    expmt_results.act_result_tbl.Experiment = ...
                repmat(expmt_dirs(i).name,  Nact_rows, 1);
    expmt_results.act_result_tbl.TissueStiffness = ...
                stiffness1*ones(Nact_rows, 1);

    expmt_results.pred_result_tbl.Experiment = ...
                repmat(expmt_dirs(i).name, Npred_rows, 1);
    expmt_results.pred_result_tbl.TissueStiffness = ...
                stiffness1*ones(Npred_rows, 1);
            
    
            
    % remove any current entries for this experiment
    act_result_tbl(strcmp(act_result_tbl.Experiment, expmt_dirs(i).name),:) = [];
    pred_result_tbl(strcmp(pred_result_tbl.Experiment, expmt_dirs(i).name),:) = [];
            
    % add to the current table
    act_result_tbl = [act_result_tbl; expmt_results.act_result_tbl];
    pred_result_tbl = [pred_result_tbl; expmt_results.pred_result_tbl];
    

    disp("Completed.");
    disp(" ");

end

%% Perform statistics
% actual results
act_result_summ_Lref = groupsummary(act_result_tbl, 'L_ref', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_InsHole = groupsummary(act_result_tbl, 'Ins_Hole', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_stiff = groupsummary(act_result_tbl, 'TissueStiffness', {'mean', 'max', 'std'},...
    act_col_err_names);

% prediction results
pred_result_tbl_nosame = pred_result_tbl(pred_result_tbl.L_ref < pred_result_tbl.L_pred,:);
pred_result_summ_Lref = groupsummary(pred_result_tbl_nosame, 'L_ref', {'mean', 'max', 'std'},...
    pred_col_err_names);
pred_result_summ_Lpred = groupsummary(pred_result_tbl_nosame, 'L_pred', {'mean', 'max', 'std'},...
    pred_col_err_names);
pred_result_summ_InsHole = groupsummary(pred_result_tbl_nosame, 'Ins_Hole', {'mean', 'max', 'std'}, ...
    pred_col_err_names);
pred_results_summ_stiff = groupsummary(pred_result_tbl_nosame, 'TissueStiffness', {'mean', 'max', 'std'},...
    pred_col_err_names); 

%% Save the Data
if save_bool
    % .mat file
    save(fullfile(save_dir, result_file), 'act_result_tbl', 'pred_result_tbl');
    fprintf("Saved data to: %s\n", fullfile(needle_dir, result_file));
    
    % tables
    writetable(act_result_tbl, fullfile(save_dir, result_excel_file),...
        'WriteVariableNames',true, 'Sheet', 'Act. All Results');
    
    writetable(pred_result_tbl, fullfile(save_dir, result_excel_file),...
        'WriteVariableNames',true, 'Sheet', 'Pred. All Results');
    fprintf("Wrote data to: %s\n", fullfile(save_dir, result_excel_file));
    
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
boxplot(act_result_tbl.RMSE, act_result_tbl.TissueStiffness);
xlabel("Tissue Stiffness (OO units)"); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% - In-Plane Error
ax_act_stiff_inplane = subplot(1,3,2);
ax_act_stiff_inplane.Position = ax_act_stiff_inplane.Position + ax_pos_adj;
boxplot(act_result_tbl.InPlane, act_result_tbl.TissueStiffness);
xlabel("Tissue Stiffness (OO units)"); ylabel("In-Plane Error(mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% - Out-of-Plane Error
ax_act_stiff_outplane = subplot(1,3,3);
ax_act_stiff_outplane.Position = ax_act_stiff_outplane.Position + ax_pos_adj;
boxplot(act_result_tbl.OutPlane, act_result_tbl.TissueStiffness);
xlabel("Tissue Stiffness (OO units)"); ylabel("Out-of-Plane Error(mm)");
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

%% Plot the Results (Prediction | FBG-FBG)
prediction_mask = pred_result_tbl.L_ref < pred_result_tbl.L_pred;

% Prediction per Insertion Hole
fig_pred_inshole = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_inshole, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_inshole_rmse = subplot(1,4,1);
ax_pred_inshole_rmse.Position = ax_pred_inshole_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_inshole_tip = subplot(1,4,2);
ax_pred_inshole_tip.Position = ax_pred_inshole_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_inshole_inplane = subplot(1,4,3);
ax_pred_inshole_inplane.Position = ax_pred_inshole_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_inshole_outplane = subplot(1,4,4);
ax_pred_inshole_outplane.Position = ax_pred_inshole_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("FBG Shape Predicted FBG Shape Analysis per Insertion Hole",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_inshole_rmse.YLim = [0, max([ax_pred_inshole_rmse.YLim, ...
                                     ax_pred_inshole_tip.YLim,...
                                     ax_pred_inshole_inplane.YLim,...
                                     ax_pred_inshole_outplane.YLim,...
                                     max_yl])];
ax_pred_inshole_tip.YLim = ax_pred_inshole_rmse.YLim;
ax_pred_inshole_inplane.YLim = ax_pred_inshole_rmse.YLim;
ax_pred_inshole_outplane.YLim = ax_pred_inshole_rmse.YLim;


% Prediction per Insertion Depth
fig_pred_Lref = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_Lref, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_Lref_rmse = subplot(1,4,1);
ax_pred_Lref_rmse.Position = ax_pred_Lref_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_Lref_tip = subplot(1,4,2);
ax_pred_Lref_tip.Position = ax_pred_Lref_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_Lref_inplane = subplot(1,4,3);
ax_pred_Lref_inplane.Position = ax_pred_Lref_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_Lref_outplane = subplot(1,4,4);
ax_pred_Lref_outplane.Position = ax_pred_Lref_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("FBG Shape Predicted FBG Shape Analysis per Insertion Depth",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_Lref_rmse.YLim = [0, max([ax_pred_Lref_rmse.YLim, ...
                                     ax_pred_Lref_tip.YLim,...
                                     ax_pred_Lref_inplane.YLim,...
                                     ax_pred_Lref_outplane.YLim,...
                                     max_yl])];
ax_pred_Lref_tip.YLim = ax_pred_Lref_rmse.YLim;
ax_pred_Lref_inplane.YLim = ax_pred_Lref_rmse.YLim;
ax_pred_Lref_outplane.YLim = ax_pred_Lref_rmse.YLim;


% Prediction per Predicted Insertion Depth
fig_pred_Lpred = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_Lpred, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_Lpred_rmse = subplot(1,4,1);
ax_pred_Lpred_rmse.Position = ax_pred_Lpred_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_Lpred_tip = subplot(1,4,2);
ax_pred_Lpred_tip.Position = ax_pred_Lpred_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_Lpred_inplane = subplot(1,4,3);
ax_pred_Lpred_inplane.Position = ax_pred_Lpred_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_Lpred_outplane = subplot(1,4,4);
ax_pred_Lpred_outplane.Position = ax_pred_Lpred_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("FBG Shape Predicted FBG Shape Analysis per Predicted Insertion Depth",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_Lpred_rmse.YLim = [0, max([ax_pred_Lpred_rmse.YLim, ...
                                     ax_pred_Lpred_tip.YLim,...
                                     ax_pred_Lpred_inplane.YLim,...
                                     ax_pred_Lpred_outplane.YLim,...
                                     max_yl])];
ax_pred_Lpred_tip.YLim = ax_pred_Lpred_rmse.YLim;
ax_pred_Lpred_inplane.YLim = ax_pred_Lpred_rmse.YLim;
ax_pred_Lpred_outplane.YLim = ax_pred_Lpred_rmse.YLim;


% Prediction per Tissue Stiffness
fig_pred_stiff = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_stiff, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_stiff_rmse = subplot(1,4,1);
ax_pred_stiff_rmse.Position = ax_pred_stiff_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_stiff_tip = subplot(1,4,2);
ax_pred_stiff_tip.Position = ax_pred_stiff_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_stiff_inplane = subplot(1,4,3);
ax_pred_stiff_inplane.Position = ax_pred_stiff_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_stiff_outplane = subplot(1,4,4);
ax_pred_stiff_outplane.Position = ax_pred_stiff_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("FBG Shape Predicted FBG Shape Analysis per Tissue Stiffness",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_stiff_rmse.YLim = [0, max([ax_pred_stiff_rmse.YLim, ...
                                     ax_pred_stiff_tip.YLim,...
                                     ax_pred_stiff_inplane.YLim,...
                                     ax_pred_stiff_outplane.YLim,...
                                     max_yl])];
ax_pred_stiff_tip.YLim = ax_pred_stiff_rmse.YLim;
ax_pred_stiff_inplane.YLim = ax_pred_stiff_rmse.YLim;
ax_pred_stiff_outplane.YLim = ax_pred_stiff_rmse.YLim;

% Prediction per Insertion Depth Increment
fig_pred_insdepth_inc = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_insdepth_inc, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_insdepth_inc_rmse = subplot(1,4,1);
ax_pred_insdepth_inc_rmse.Position = ax_pred_insdepth_inc_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_insdepth_inc_tip = subplot(1,4,2);
ax_pred_insdepth_inc_tip.Position = ax_pred_insdepth_inc_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_insdepth_inc_inplane = subplot(1,4,3);
ax_pred_insdepth_inc_inplane.Position = ax_pred_insdepth_inc_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_insdepth_inc_outplane = subplot(1,4,4);
ax_pred_insdepth_inc_outplane.Position = ax_pred_insdepth_inc_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("Stereo Reconstruction Predicted FBG Shape Analysis per Insertion Depth Increment",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_insdepth_inc_rmse.YLim = [0, max([ax_pred_insdepth_inc_rmse.YLim, ...
                                     ax_pred_insdepth_inc_tip.YLim,...
                                     ax_pred_insdepth_inc_inplane.YLim,...
                                     ax_pred_insdepth_inc_outplane.YLim,...
                                     max_yl])];
ax_pred_insdepth_inc_tip.YLim = ax_pred_insdepth_inc_rmse.YLim;
ax_pred_insdepth_inc_inplane.YLim = ax_pred_insdepth_inc_rmse.YLim;
ax_pred_insdepth_inc_outplane.YLim = ax_pred_insdepth_inc_rmse.YLim;

%% Plot the Results (Prediction | FBG-Cam)
prediction_mask = pred_result_tbl.L_ref < pred_result_tbl.L_pred;

% Prediction per Insertion Hole
fig_pred_cam_inshole = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_cam_inshole, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_cam_inshole_rmse = subplot(1,4,1);
ax_pred_cam_inshole_rmse.Position = ax_pred_cam_inshole_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_RMSE(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_cam_inshole_tip = subplot(1,4,2);
ax_pred_cam_inshole_tip.Position = ax_pred_cam_inshole_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_TipError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_cam_inshole_inplane = subplot(1,4,3);
ax_pred_cam_inshole_inplane.Position = ax_pred_cam_inshole_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_InPlaneError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_cam_inshole_outplane = subplot(1,4,4);
ax_pred_cam_inshole_outplane.Position = ax_pred_cam_inshole_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_OutPlaneError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("Stereo Reconstruction Predicted FBG Shape Analysis per Insertion Hole",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_cam_inshole_rmse.YLim = [0, max([ax_pred_cam_inshole_rmse.YLim, ...
                                     ax_pred_cam_inshole_tip.YLim,...
                                     ax_pred_cam_inshole_inplane.YLim,...
                                     ax_pred_cam_inshole_outplane.YLim,...
                                     max_yl])];
ax_pred_cam_inshole_tip.YLim = ax_pred_cam_inshole_rmse.YLim;
ax_pred_cam_inshole_inplane.YLim = ax_pred_cam_inshole_rmse.YLim;
ax_pred_cam_inshole_outplane.YLim = ax_pred_cam_inshole_rmse.YLim;


% Prediction per Insertion Depth
fig_pred_cam_Lref = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_cam_Lref, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_cam_Lref_rmse = subplot(1,4,1);
ax_pred_cam_Lref_rmse.Position = ax_pred_cam_Lref_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_RMSE(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_cam_Lref_tip = subplot(1,4,2);
ax_pred_cam_Lref_tip.Position = ax_pred_cam_Lref_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_TipError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_cam_Lref_inplane = subplot(1,4,3);
ax_pred_cam_Lref_inplane.Position = ax_pred_cam_Lref_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_InPlaneError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_cam_Lref_outplane = subplot(1,4,4);
ax_pred_cam_Lref_outplane.Position = ax_pred_cam_Lref_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_OutPlaneError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth (mm)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("Stereo Reconstruction Predicted FBG Shape Analysis per Insertion Depth",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_cam_Lref_rmse.YLim = [0, max([ax_pred_cam_Lref_rmse.YLim, ...
                                     ax_pred_cam_Lref_tip.YLim,...
                                     ax_pred_cam_Lref_inplane.YLim,...
                                     ax_pred_cam_Lref_outplane.YLim,...
                                     max_yl])];
ax_pred_cam_Lref_tip.YLim = ax_pred_cam_Lref_rmse.YLim;
ax_pred_cam_Lref_inplane.YLim = ax_pred_cam_Lref_rmse.YLim;
ax_pred_cam_Lref_outplane.YLim = ax_pred_cam_Lref_rmse.YLim;


% Prediction per Predicted Insertion Depth
fig_pred_cam_Lpred = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_cam_Lpred, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_cam_Lpred_rmse = subplot(1,4,1);
ax_pred_cam_Lpred_rmse.Position = ax_pred_cam_Lpred_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_RMSE(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_cam_Lpred_tip = subplot(1,4,2);
ax_pred_cam_Lpred_tip.Position = ax_pred_cam_Lpred_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_TipError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_cam_Lpred_inplane = subplot(1,4,3);
ax_pred_cam_Lpred_inplane.Position = ax_pred_cam_Lpred_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_InPlaneError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_cam_Lpred_outplane = subplot(1,4,4);
ax_pred_cam_Lpred_outplane.Position = ax_pred_cam_Lpred_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_OutPlaneError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('Predicted Insertion Depth (mm)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("Stereo Reconstruction Predicted FBG Shape Analysis per Predicted Insertion Depth",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_cam_Lpred_rmse.YLim = [0, max([ax_pred_cam_Lpred_rmse.YLim, ...
                                     ax_pred_cam_Lpred_tip.YLim,...
                                     ax_pred_cam_Lpred_inplane.YLim,...
                                     ax_pred_cam_Lpred_outplane.YLim,...
                                     max_yl])];
ax_pred_cam_Lpred_tip.YLim = ax_pred_cam_Lpred_rmse.YLim;
ax_pred_cam_Lpred_inplane.YLim = ax_pred_cam_Lpred_rmse.YLim;
ax_pred_cam_Lpred_outplane.YLim = ax_pred_cam_Lpred_rmse.YLim;


% Prediction per Tissue Stiffness
fig_pred_cam_stiff = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_cam_stiff, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_cam_stiff_rmse = subplot(1,4,1);
ax_pred_cam_stiff_rmse.Position = ax_pred_cam_stiff_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_RMSE(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_cam_stiff_tip = subplot(1,4,2);
ax_pred_cam_stiff_tip.Position = ax_pred_cam_stiff_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_TipError(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_cam_stiff_inplane = subplot(1,4,3);
ax_pred_cam_stiff_inplane.Position = ax_pred_cam_stiff_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_InPlaneError(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_cam_stiff_outplane = subplot(1,4,4);
ax_pred_cam_stiff_outplane.Position = ax_pred_cam_stiff_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_OutPlaneError(prediction_mask), pred_result_tbl.TissueStiffness(prediction_mask));
xlabel('Tissue Stiffness (OO units)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("Stereo Reconstruction Predicted FBG Shape Analysis per Tissue Stiffness",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_cam_stiff_rmse.YLim = [0, max([ax_pred_cam_stiff_rmse.YLim, ...
                                     ax_pred_cam_stiff_tip.YLim,...
                                     ax_pred_cam_stiff_inplane.YLim,...
                                     ax_pred_cam_stiff_outplane.YLim,...
                                     max_yl])];
ax_pred_cam_stiff_tip.YLim = ax_pred_cam_stiff_rmse.YLim;
ax_pred_cam_stiff_inplane.YLim = ax_pred_cam_stiff_rmse.YLim;
ax_pred_cam_stiff_outplane.YLim = ax_pred_cam_stiff_rmse.YLim;

% Prediction per Insertion Depth Increment
fig_pred_cam_insdepth_inc = figure(fig_counter);
fig_counter = fig_counter + 1;
set(fig_pred_cam_insdepth_inc, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% - RMSE
ax_pred_cam_insdepth_inc_rmse = subplot(1,4,1);
ax_pred_cam_insdepth_inc_rmse.Position = ax_pred_cam_insdepth_inc_rmse.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_RMSE(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% - Tip Error
ax_pred_cam_insdepth_inc_tip = subplot(1,4,2);
ax_pred_cam_insdepth_inc_tip.Position = ax_pred_cam_insdepth_inc_tip.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_TipError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% - In-Plane Errors
ax_pred_cam_insdepth_inc_inplane = subplot(1,4,3);
ax_pred_cam_insdepth_inc_inplane.Position = ax_pred_cam_insdepth_inc_inplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_InPlaneError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);

% - Out-of-Plane Errors
ax_pred_cam_insdepth_inc_outplane = subplot(1,4,4);
ax_pred_cam_insdepth_inc_outplane.Position = ax_pred_cam_insdepth_inc_outplane.Position + ax_pos_adj;
boxplot(pred_result_tbl.Cam_OutPlaneError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% - axis limits and titling
sgtitle("Stereo Reconstruction Predicted FBG Shape Analysis per Insertion Depth Increment",...
    'Fontsize', 22, 'fontweight', 'bold');
ax_pred_cam_insdepth_inc_rmse.YLim = [0, max([ax_pred_cam_insdepth_inc_rmse.YLim, ...
                                     ax_pred_cam_insdepth_inc_tip.YLim,...
                                     ax_pred_cam_insdepth_inc_inplane.YLim,...
                                     ax_pred_cam_insdepth_inc_outplane.YLim,...
                                     max_yl])];
ax_pred_cam_insdepth_inc_tip.YLim = ax_pred_cam_insdepth_inc_rmse.YLim;
ax_pred_cam_insdepth_inc_inplane.YLim = ax_pred_cam_insdepth_inc_rmse.YLim;
ax_pred_cam_insdepth_inc_outplane.YLim = ax_pred_cam_insdepth_inc_rmse.YLim;

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
        
    % Predicted Results: FBG-FBG
    savefigas(fig_pred_inshole, strcat(dataout_dir_file, '_fbg-predicted-error_inshole'),...
        'Verbose', true);
        
    savefigas(fig_pred_Lref, strcat(dataout_dir_file, '_fbg-predicted-error_Lref'),...
        'Verbose', true);
        
    savefigas(fig_pred_Lpred, strcat(dataout_dir_file, '_fbg-predicted-error_Lpred'),...
        'Verbose', true);
        
    savefigas(fig_pred_stiff, strcat(dataout_dir_file, '_fbg-predicted-error_stiff'),...
        'Verbose', true);
    
    savefigas(fig_pred_insdepth_inc, strcat(dataout_dir_file, '_fbg-predicted-error_insdepth_inc'),...
        'Verbose', true);
    
    % Predicted Results: FBG-Cam
    savefigas(fig_pred_cam_inshole, strcat(dataout_dir_file, '_cam-predicted-error_inshole'),...
        'Verbose', true);
        
    savefigas(fig_pred_cam_Lref, strcat(dataout_dir_file, '_cam-predicted-error_Lref'),...
        'Verbose', true);
        
    savefigas(fig_pred_cam_Lpred, strcat(dataout_dir_file, '_cam-predicted-error_Lpred'),...
        'Verbose', true);
        
    savefigas(fig_pred_cam_stiff, strcat(dataout_dir_file, '_cam-predicted-error_stiff'),...
        'Verbose', true);
    
    savefigas(fig_pred_cam_insdepth_inc, strcat(dataout_dir_file, '_cam-predicted-error_insdepth_inc'),...
        'Verbose', true);
        
end

%% Termination
disp("Press [ENTER] to finish the program");
pause;
close all;
disp("Program Completed.");
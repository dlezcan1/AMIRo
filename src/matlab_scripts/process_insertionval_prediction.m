%% process_insertionval_prediction
%
% script to process insertion validation prediction errors
%
% - written by: Dimitri Lezcano

%% Set-up
clear;
configure_env on;
set(0,'DefaultAxesFontSize',18)

% directories to iterate through
expmt_dir = "../../data/3CH-4AA-0004/08-30-2021_Insertion-Expmt-1/";
trial_dirs = dir(expmt_dir + "Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% predicted insertion params
L_preds = 105:5:120; % static method
p = 0.592;
ds = 0.5;

% weighted FBG measurement options
use_weights = true;

% data file
data_file = "FBGdata_3d-params.txt";
camera_pos_file = "left-right_3d-pts.csv";

% saving options
save_bool = true;
dataout_file = "FBGdata_prediction";
if use_weights
    dataout_file = strcat(dataout_file, "_FBG-weights");
    data_file = "FBGdata_FBG-weights_3d-params.txt";
end

% loading options
recalculate = false; % whether to recalculate predictions and errors

% needle parameters
needleparams = load("../../shape-sensing/shapesensing_needle_properties_18G.mat");

%% Iterate over the trial directories
dir_prev = "";
hole_num_prev = 0;

% set-up result tables 
% - non-prediction table
act_col_names = {'Ins_Hole', 'L_ref', ...
             'cam_shape', 'fbg_shape', 'RMSE', 'MaxError', 'InPlane', 'OutPlane'};
Nact_cols = numel(act_col_names);
Nact_err  = Nact_cols - find(strcmp(act_col_names, 'fbg_shape'));
act_col_err_names = act_col_names(end-Nact_err+1:end);
actcol_types = cat(2, {'uint8'}, repmat({'double'}, 1, Nact_cols-1));
actcol_units = cat(2, {''}, repmat({'mm'}, 1, Nact_cols-1));

% - prediction table         
pred_col_names = {'Ins_Hole', 'L_ref', 'L_pred', ...
             'cam_shape', 'fbgref_shape', 'pred_shape', 'q_optim', ...
             'FBG_RMSE', 'FBG_MaxError', 'FBG_TipError', 'FBG_InPlaneError', 'FBG_OutPlaneError', ...
             'Cam_RMSE', 'Cam_MaxError', 'Cam_TipError', 'Cam_InPlaneError', 'Cam_OutPlaneError'};
Npred_cols = numel(pred_col_names);
Npred_err = Npred_cols - find(strcmp(pred_col_names, "q_optim")); % number of error columns
pred_col_err_names = pred_col_names(end-Npred_err+1:end);
predcol_types = cat(2, {'uint8'}, repmat({'double'}, 1, Npred_cols-1));
predcol_units = {'', 'mm', 'mm', 'mm', 'mm', 'mm', ''};
predcol_units = cat(2, predcol_units, repmat({'mm'}, 1, Npred_err));

% generate the result tables
% - actual results
act_result_tbl = table('Size', [0,Nact_cols], 'VariableTypes', actcol_types, ...
        'VariableNames', act_col_names);
act_result_tbl.Properties.VariableUnits = actcol_units;
act_results_tbl.Properties.Description = "Comparison between Actual FBG shapes and stereo reconstruction";

% - prediction results
pred_result_tbl = table('Size', [0,Npred_cols], 'VariableTypes', predcol_types, ...
        'VariableNames', pred_col_names);
pred_result_tbl.Properties.VariableUnits = predcol_units;
pred_results_tbl.Properties.Description = "Comparison between Actual FBG shapes, stereo reconstruction, and predicted FBG shapes.";

% iterate through all of the trials compiling the data
if ~isfile(fullfile(expmt_dir, strcat(dataout_file, '_results.mat'))) || recalculate 
    progressbar('Compile Data...');
    for i = 1:length(trial_dirs)   
        % trial operations
        L = str2double(trial_dirs(i).name);
        re_ret = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
        hole_num = str2double(re_ret{1}{1});

        % get the files
        d = fullfile(trial_dirs(i).folder, trial_dirs(i).name);
        fbg_paramin_file = fullfile(d, data_file);
        cam_shapein_file = fullfile(d, camera_pos_file);

        % load the needle shape param file
        [kc, w_init, ~, theta0] = readin_fbgparam_file(fbg_paramin_file);

        % generate the reference shape
        [~, p_fbg_ref] = fn_intgEP_1layer_Dimitri(kc, w_init, theta0, L, 0, ds, ...
                    needleparams.B, needleparams.Binv);

        % get the camera positions
        p_cam = readmatrix(cam_shapein_file)';
        p_cam = p_cam(1:3,:);

        % compute actual FBG and Camera errors
        [cam_ref_errors,p_cam_interp_tf] = compute_camera_errors(p_cam, p_fbg_ref, ds);

        % append results to actual result table
        error_init_vals = num2cell(zeros(1,Nact_err));
        act_result_tbl = [act_result_tbl;
                          {hole_num, L, p_cam_interp_tf, p_fbg_ref, ...
                          cam_ref_errors.RMSE, cam_ref_errors.Max, ...
                          cam_ref_errors.In_Plane, cam_ref_errors.Out_Plane}];

        % predict the insertion shapes
        [p_preds, ~, ~, q_best] = predict_insertion_singlelayer(L_preds, L, kc, w_init, needleparams, ...
                p, 'optim_lb', 0.0, 'optim_ub', 1.0, 'theta0', theta0, ...
                'optim_display', 'none');

        if ~iscell(p_preds) % cell-ify p_preds
            p_preds = {p_preds};
        end

        % compute errors and append to prediction table
        for j = 1:numel(p_preds)
            error_init_vals = num2cell(zeros(1,Npred_err));
            % append to table        
            if L == L_preds(j)
                pred_result_tbl = [pred_result_tbl;
                              {hole_num, L, L_preds(j), p_cam_interp_tf, p_fbg_ref, p_preds{j}, ...
                               q_best, error_init_vals{:}}];
            else
                pred_result_tbl = [pred_result_tbl;
                              {hole_num, L, L_preds(j), zeros(3,0), zeros(3,0), p_preds{j}, ...
                               q_best, error_init_vals{:}}];
            end

        end

        % generate the insertion prediction error at the end of the insertion hole
        if hole_num_prev > 0 && hole_num_prev ~= hole_num
            fprintf("Completed Insertion Hole #%d.\n", hole_num_prev);
        end

        % update previous hole number
        hole_num_prev = hole_num;
        progressbar(i/length(trial_dirs));

    end
    fprintf("Completed Insertion Hole #%d.\n\n", hole_num_prev);
else
    l = load(fullfile(expmt_dir, strcat(dataout_file, '_results.mat')));
    act_result_tbl = l.act_result_tbl;
    pred_result_tbl = l.pred_result_tbl;
end

%% Compute errors and fill out table
hole_nums = reshape(unique(pred_result_tbl.Ins_Hole), 1, []);

% iterate through the insertion hole numbers
if ~isfile(fullfile(expmt_dir, strcat(dataout_file, '_results.mat'))) || recalculate 
    for hole_num = hole_nums
        fprintf("Computing errors for insertion hole #%d...\n", hole_num);
        subtbl = pred_result_tbl(pred_result_tbl.Ins_Hole == hole_num, :);

        % add in all of the reference shapes
        for L_ref = L_preds
            subtbl_ref = subtbl(subtbl.L_pred == L_ref,:);
            p_fbg_ref = subtbl_ref(subtbl_ref.L_ref == L_ref, :).fbgref_shape{1};
            p_cam = subtbl_ref(subtbl_ref.L_ref == L_ref, :).cam_shape{1};

            for j = 1:size(subtbl_ref,1)
                p_pred = subtbl_ref{j,'pred_shape'}{1};
                N_overlap = min(size(p_cam,2), size(p_pred,2));

                % compute errors
                errors_fbg = compute_errors(p_fbg_ref, p_pred);
                errors_cam = compute_errors(p_cam(:,end-N_overlap+1:end), ...
                                            p_pred(:,end-N_overlap+1:end));

                % update table with errors and fbgref and cam shapes
                subtbl_ref{j,'cam_shape'}{1} = p_cam;
                subtbl_ref{j,'fbgref_shape'}{1} = p_fbg_ref;

                subtbl_ref{j,'Cam_RMSE'} = errors_cam.RMSE;
                subtbl_ref{j,'Cam_MaxError'} = errors_cam.Max;
                subtbl_ref{j,'Cam_TipError'} = errors_cam.Tip;
                subtbl_ref{j,'Cam_InPlaneError'} = errors_cam.In_Plane;
                subtbl_ref{j,'Cam_OutPlaneError'} = errors_cam.Out_Plane;

                subtbl_ref{j,'FBG_RMSE'} = errors_fbg.RMSE;
                subtbl_ref{j,'FBG_MaxError'} = errors_fbg.Max;
                subtbl_ref{j,'FBG_TipError'} = errors_fbg.Tip;
                subtbl_ref{j,'FBG_InPlaneError'} = errors_fbg.In_Plane;
                subtbl_ref{j,'FBG_OutPlaneError'} = errors_fbg.Out_Plane;
            end

            % update subtbl with subtbl_ref
            subtbl(subtbl.L_pred == L_ref,:) = subtbl_ref;
        end

        % update result_tbl with subtbl
        pred_result_tbl(pred_result_tbl.Ins_Hole == hole_num,:) = subtbl;

    end
end

%% Summarization
% - actual FBG summarization
act_result_summ_Lref = groupsummary(act_result_tbl, 'L_ref', {'mean', 'max', 'std'},...
    act_col_err_names);
act_result_summ_InsHole = groupsummary(act_result_tbl, 'Ins_Hole', {'mean', 'max', 'std'},...
    act_col_err_names);

disp("Actual FBG Summary: Per Reference Insertion Length");
disp(act_result_summ_Lref);
disp(" ");

disp("Actual FBG Summary: Per Reference Insertion Hole");
disp(act_result_summ_InsHole);
disp(" ");

% - prediction summarization
pred_result_tbl_nosame = pred_result_tbl(pred_result_tbl.L_ref ~= pred_result_tbl.L_pred,:);
pred_result_summ_Lref = groupsummary(pred_result_tbl_nosame, 'L_ref', {'mean', 'max', 'std'},...
    pred_col_err_names);
pred_result_summ_Lpred = groupsummary(pred_result_tbl_nosame, 'L_pred', {'mean', 'max', 'std'},...
    pred_col_err_names);
pred_result_summ_InsHole = groupsummary(pred_result_tbl_nosame, 'Ins_Hole', {'mean', 'max', 'std'}, ...
    pred_col_err_names);

disp("Prediction Summary: Per Reference Insertion Length");
disp(pred_result_summ_Lref);
disp(" ");

disp("Prediction Summary: Per Predicted Insertion Length");
disp(pred_result_summ_Lpred);
disp(" ");

disp("Prediction Summary: Per Insertion Hole");
disp(pred_result_summ_InsHole);
disp(" ");

if save_bool
    result_tbl_file = fullfile(expmt_dir, strcat(dataout_file, "_results"));
    result_tbl_file_xlsx = strcat(result_tbl_file, ".xlsx");
    result_tbl_file_mat = strcat(result_tbl_file, ".mat");

    % write entire results
    writetable(removevars(act_result_tbl, {'cam_shape', 'fbg_shape'}),...
        result_tbl_file_xlsx,'WriteVariableNames', true, 'Sheet', 'Act. All Results');
    writetable(removevars(pred_result_tbl, {'cam_shape', 'fbgref_shape', 'pred_shape'}),...
       result_tbl_file_xlsx, 'WriteVariableNames', true, 'Sheet', 'Pred. All Results');

    % write actual FBG summary results
    writetable(act_result_summ_Lref, result_tbl_file_xlsx, 'WriteVariablenames', true, ...
        'Sheet', 'Act. Summary Lref');
    writetable(act_result_summ_InsHole, result_tbl_file_xlsx, 'WriteVariablenames', true, ...
        'Sheet', 'Act. Summary Insertion Hole');
    
    % write prediction summary results
    writetable(pred_result_summ_Lref, result_tbl_file_xlsx, 'WriteVariableNames', true, ...
        'Sheet', 'Pred. Summary Lref');
    writetable(pred_result_summ_Lpred, result_tbl_file_xlsx, 'WriteVariableNames', true, ...
        'Sheet', 'Pred. Summary Lpred');
    writetable(pred_result_summ_InsHole, result_tbl_file_xlsx, 'WriteVariableNames', true, ...
        'Sheet', 'Pred. Summary Insertion Hole');
    
    % display out
    fprintf("Wrote Results to: %s\n", result_tbl_file_xlsx);
    
    % save table to mat file
    save(result_tbl_file_mat, 'act_result_tbl', 'pred_result_tbl', ...
        'act_result_summ_Lref', 'pred_result_summ_Lref', ...
        'pred_result_summ_Lpred',...
        'act_result_summ_InsHole', 'pred_result_summ_InsHole');
    
    % display out 
    fprintf("Wrote Results to: %s\n", result_tbl_file_mat);
    disp(" ");
end
%% Plotting of Needle Shapes
if ~isfile(fullfile(expmt_dir, strcat(dataout_file, '_results.mat'))) || recalculate
    fig = figure(1);
    set(fig, 'units', 'normalized', 'Position', [0.05,0.05,0.9,0.85])
    % iterate over each insertion hole
    for hole_num = hole_nums
        subtbl_hole = pred_result_tbl(pred_result_tbl.Ins_Hole == hole_num,:);

        for i = 1:numel(L_preds)
            L_i = L_preds(i);

            % grab all of the predicted insertion lengths
            subtbl_hole_Li = subtbl_hole(subtbl_hole.L_pred == L_i,:);
            pos_cam = subtbl_hole_Li{1,'cam_shape'}{1};

            subplot(2,1,1);
            plot(pos_cam(3,:), pos_cam(1,:), 'DisplayName', 'Stereo Recons.', 'LineWidth', 4); hold on;

            subplot(2,1,2);
            plot(pos_cam(3,:), pos_cam(2,:), 'DisplayName', 'Stereo Recons.', 'LineWidth', 4); hold on;

            % iterate through each length
            for j = 1:size(subtbl_hole_Li,1)
                pos = subtbl_hole_Li(j,:).pred_shape{1};
                switch sign(subtbl_hole_Li.L_pred(j) - subtbl_hole_Li.L_ref(j))
                    case 1 % L_pred > L_ref
                        lbl = sprintf("FWD Pred. from %.0f mm ", subtbl_hole_Li.L_ref(j));

                    case -1 % L_ref > L_pred
                        lbl = sprintf("BWD Pred. from %.0f mm", subtbl_hole_Li.L_ref(j));

                    case 0 % L_ref = L_pred
                        lbl = "Actual FBG Shape";
                        pos = subtbl_hole_Li{j,'fbgref_shape'}{1};
                end

                % plot the 3D shape
                subplot(2,1,1);
                plot(pos(3,:), pos(1,:), 'DisplayName', lbl, 'LineWidth', 2); hold on;

                subplot(2,1,2);
                plot(pos(3,:), pos(2,:), 'DisplayName', lbl, 'LineWidth', 2); hold on;

            end

            subplot(2,1,1);
            hold off;
            legend('Location', 'northwest');
            axis equal; grid on;
            xlabel('z (mm)'); ylabel('x (mm)');
            title(sprintf("Predicted FBG Shape | Insertion Hole #%d | L = %.1f mm", ...
                    hole_num, L_i))


            subplot(2,1,2);
            hold off;
            axis equal; grid on;
            xlabel('z (mm)'); ylabel('y (mm)');
            axis equal; grid on;

            if save_bool
               save_dir = fullfile(expmt_dir, sprintf('Insertion%d/',hole_num));
               figfile_savebase = fullfile(save_dir, sprintf(strcat(dataout_file,...
                   '_predicted-shape-comp_Lpred-%.1f'),L_i));

               savefigas(fig, figfile_savebase, 'Verbose', true);
               
            end

            pause(1);
        end
    end
end

%% Error Plotting
% Actual Error Plotting
pos_adj = [0, 0, 0, -0.05]; % adjust the height of the position
max_yl = 1.25;

% - Per insertion holegit
fig_inshole_errors = figure(2);
set(fig_inshole_errors, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% -- RMSE Errors
ax1_inshole = subplot(1,3,1);
ax1_inshole.Position = ax1_inshole.Position + pos_adj;
boxplot(act_result_tbl.RMSE, act_result_tbl.Ins_Hole);
xlabel('Insertion Hole Number'); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% -- In-Plane Errors
ax2_inshole = subplot(1,3,2);
ax2_inshole.Position = ax2_inshole.Position + pos_adj;
boxplot(act_result_tbl.InPlane, act_result_tbl.Ins_Hole);
xlabel('Insertion Hole Number'); ylabel("In-Plane Error (mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% -- Out-of-Plane Errors
ax3_inshole = subplot(1,3,3);
ax3_inshole.Position = ax3_inshole.Position + pos_adj;
boxplot(act_result_tbl.OutPlane, act_result_tbl.Ins_Hole);
xlabel('Insertion Hole Number'); ylabel("Out-of-Plane Error (mm)");
title("Mean Out-of-Plane Errors", 'fontsize', 20);

sgtitle("Errors between Stereo and FBG Reconstructed Shapes per Insertion Hole",...
    'FontSize', 22, 'FontWeight', 'bold')

% -- set axes limits
ax1_inshole.YLim = [0, max([ax1_inshole.YLim, ax2_inshole.YLim, ax3_inshole.YLim, max_yl])];
ax2_inshole.YLim = ax1_inshole.YLim;
ax3_inshole.YLim = ax1_inshole.YLim;


% Errors per Insertion Depth
fig_Lref_errors = figure(3);
set(fig_Lref_errors, 'units', 'normalized', 'position',[ -1, 0.0370, 1.0000, 0.8917 ]);
% -- RMSE Errors
ax1_Lref = subplot(1,3,1);
ax1_Lref.Position = ax1_Lref.Position + pos_adj;
boxplot(act_result_tbl.RMSE, act_result_tbl.L_ref);
% hold on; yline(0.5, 'r--', 'DisplayName', '0.5 mm Error'); hold off;
legend()
xlabel('Insertion Depth (mm)'); ylabel("RMSE (mm)");
title("RMSE", 'fontsize', 20);

% -- In-Plane Errors
ax2_Lref = subplot(1,3,2);
ax2_Lref.Position = ax2_Lref.Position + pos_adj;
boxplot(act_result_tbl.InPlane, act_result_tbl.L_ref);
xlabel('Insertion Depth (mm)'); ylabel("In-Plane Error (mm)");
title("Mean In-Plane Errors", 'fontsize', 20);

% -- Out-of-Plane Errors
ax3_Lref = subplot(1,3,3);
ax3_Lref.Position = ax3_Lref.Position + pos_adj;
boxplot(act_result_tbl.OutPlane, act_result_tbl.L_ref);
xlabel('Insertion Depth (mm)'); ylabel("Out-of-Plane Error (mm)");
title("Mean Out-of-Plane Errors", 'fontsize', 20);

sgtitle("Errors between Stereo and FBG Reconstructed Shapes per Insertion Depth",...
    'FontSize', 22, 'FontWeight', 'bold')

% -- set axes limits
ax1_Lref.YLim = [0, max([ax1_Lref.YLim, ax2_Lref.YLim, ax3_Lref.YLim, max_yl])];
ax2_Lref.YLim = ax1_Lref.YLim;
ax3_Lref.YLim = ax1_Lref.YLim;

% Predicted Error Plotting (FBG-FBG)
prediction_mask = pred_result_tbl.L_ref < pred_result_tbl.L_pred;
% - Per Insertion Hole
fig_inshole_prederrors = figure(4);
set(fig_inshole_prederrors, 'units', 'normalized', 'position',[ -1, 0.0370, 1.0000, 0.8917 ]);

% -- RMSE
ax1_inshole_pred = subplot(1,4,1);
ax1_inshole_pred.Position = ax1_inshole_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('RMSE (mm)'); title('RMSE', 'fontsize', 20);

% -- Tip Errors
ax2_inshole_pred = subplot(1,4,2);
ax2_inshole_pred.Position = ax2_inshole_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Tip Error (mm)'); title('Tip Error', 'fontsize', 20);

% -- In-Plane Errors
ax3_inshole_pred = subplot(1,4,3);
ax3_inshole_pred.Position = ax3_inshole_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Avg. In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);


% -- Out-of-Plane Errors
ax4_inshole_pred = subplot(1,4,4);
ax4_inshole_pred.Position = ax4_inshole_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.Ins_Hole(prediction_mask));
xlabel('Insertion Hole'); ylabel('Avg. Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% -- axis limits and titling
sgtitle("Predicted FBG Reconstructed Shapes per Insertion Hole",...
    'Fontsize', 22, 'fontweight', 'bold');
ax1_inshole_pred.YLim = [0, max([ax1_inshole_pred.YLim, ax2_inshole_pred.YLim,...
                                 ax3_inshole_pred.YLim, ax4_inshole_pred.YLim,...
                                 max_yl])];
ax2_inshole_pred.YLim = ax1_inshole_pred.YLim;
ax3_inshole_pred.YLim = ax1_inshole_pred.YLim;
ax4_inshole_pred.YLim = ax1_inshole_pred.YLim;

% - Per Insertion Depth
fig_Lref_prederrors = figure(5);
set(fig_Lref_prederrors, 'units', 'normalized', 'position',[ 0, 0.0370, 1.0000, 0.8917 ]);

% -- RMSE
ax1_Lref_pred = subplot(1,4,1);
ax1_Lref_pred.Position = ax1_Lref_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('L_{ref.} (mm)'); ylabel('RMSE (mm)'); title('RMSE', 'fontsize', 20);

% -- Tip Errors
ax2_Lref_pred = subplot(1,4,2);
ax2_Lref_pred.Position = ax2_Lref_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('L_{ref.} (mm)'); ylabel('Tip Error (mm)'); title('Tip Error', 'fontsize', 20);

% -- In-Plane Errors
ax3_Lref_pred = subplot(1,4,3);
ax3_Lref_pred.Position = ax3_Lref_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('L_{ref.} (mm)'); ylabel('Avg. In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);


% -- Out-of-Plane Errors
ax4_Lref_pred = subplot(1,4,4);
ax4_Lref_pred.Position = ax4_Lref_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.L_ref(prediction_mask));
xlabel('L_{ref.} (mm)'); ylabel('Avg. Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% -- axis limits and titling
sgtitle("Predicted FBG Reconstructed Shapes per Insertion Depth",...
    'Fontsize', 22, 'fontweight', 'bold');
ax1_Lref_pred.YLim = [0, max([ax1_Lref_pred.YLim, ax2_Lref_pred.YLim,...
                                 ax3_Lref_pred.YLim, ax4_Lref_pred.YLim,...
                                 max_yl])];
ax2_Lref_pred.YLim = ax1_Lref_pred.YLim;
ax3_Lref_pred.YLim = ax1_Lref_pred.YLim;
ax4_Lref_pred.YLim = ax1_Lref_pred.YLim;

% - Per Predicted Insertion Depth
fig_Lpred_prederrors = figure(6);
set(fig_Lpred_prederrors, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% -- RMSE
ax1_Lpred_pred = subplot(1,4,1);
ax1_Lpred_pred.Position = ax1_Lpred_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('L_{pred.} (mm)'); ylabel('RMSE (mm)'); title('RMSE', 'fontsize', 20);

% -- Tip Errors
ax2_Lpred_pred = subplot(1,4,2);
ax2_Lpred_pred.Position = ax2_Lpred_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('L_{pred.} (mm)'); ylabel('Tip Error (mm)'); title('Tip Error', 'fontsize', 20);

% -- In-Plane Errors
ax3_Lpred_pred = subplot(1,4,3);
ax3_Lpred_pred.Position = ax3_Lpred_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('L_{pred.} (mm)'); ylabel('Avg. In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);


% -- Out-of-Plane Errors
ax4_Lpred_pred = subplot(1,4,4);
ax4_Lpred_pred.Position = ax4_Lpred_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), pred_result_tbl.L_pred(prediction_mask));
xlabel('L_{pred.} (mm)'); ylabel('Avg. Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% -- axis limits and titling
sgtitle("Predicted FBG Reconstructed Shapes per Predicted Insertion Depth",...
    'Fontsize', 22, 'fontweight', 'bold');
ax1_Lpred_pred.YLim = [0, max([ax1_Lpred_pred.YLim, ax2_Lpred_pred.YLim,...
                                 ax3_Lpred_pred.YLim, ax4_Lpred_pred.YLim,...
                                 max_yl])];
ax2_Lpred_pred.YLim = ax1_Lpred_pred.YLim;
ax3_Lpred_pred.YLim = ax1_Lpred_pred.YLim;
ax4_Lpred_pred.YLim = ax1_Lpred_pred.YLim;

% - Per Insertion Depth increase
fig_Ldiff_prederrors = figure(7);
set(fig_Ldiff_prederrors, 'units', 'normalized', 'position',[ 1, 0.0370, 1.0000, 0.8917 ]);

% -- RMSE
ax1_Ldiff_pred = subplot(1,4,1);
ax1_Ldiff_pred.Position = ax1_Ldiff_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_RMSE(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('RMSE (mm)'); 
title('RMSE', 'fontsize', 20);

% -- Tip Errors
ax2_Ldiff_pred = subplot(1,4,2);
ax2_Ldiff_pred.Position = ax2_Ldiff_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_TipError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Tip Error (mm)'); 
title('Tip Error', 'fontsize', 20);

% -- In-Plane Errors
ax3_Ldiff_pred = subplot(1,4,3);
ax3_Ldiff_pred.Position = ax3_Ldiff_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_InPlaneError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Avg. In-Plane Error (mm)'); 
title('Average In-Plane Error', 'fontsize', 20);


% -- Out-of-Plane Errors
ax4_Ldiff_pred = subplot(1,4,4);
ax4_Ldiff_pred.Position = ax4_Ldiff_pred.Position + pos_adj;
boxplot(pred_result_tbl.FBG_OutPlaneError(prediction_mask), ...
    pred_result_tbl.L_pred(prediction_mask) - pred_result_tbl.L_ref(prediction_mask));
xlabel('Insertion Depth Increment (mm)'); ylabel('Avg. Out-of-Plane Error (mm)'); 
title('Average Out-of-Plane Error', 'fontsize', 20);

% -- axis limits and titling
sgtitle("Predicted FBG Reconstructed Shapes per Predicted Insertion Depth Increment",...
    'Fontsize', 22, 'fontweight', 'bold');
ax1_Ldiff_pred.YLim = [0, max([ax1_Ldiff_pred.YLim, ax2_Ldiff_pred.YLim,...
                                 ax3_Ldiff_pred.YLim, ax4_Ldiff_pred.YLim,...
                                 max_yl])];
ax2_Ldiff_pred.YLim = ax1_Ldiff_pred.YLim;
ax3_Ldiff_pred.YLim = ax1_Ldiff_pred.YLim;
ax4_Ldiff_pred.YLim = ax1_Ldiff_pred.YLim;

%% Saving
if save_bool
    % actual
    savefigas(fig_inshole_errors, fullfile(expmt_dir, strcat(dataout_file, '_insertion-hole_errors')));
        
    savefigas(fig_Lref_errors, fullfile(expmt_dir, strcat(dataout_file, '_Lref_errors')));
        
    % prediction 
    savefigas(fig_inshole_prederrors, fullfile(expmt_dir, strcat(dataout_file, '_prediction_inshole_errors')),...
        'Verbose', true);
        
    savefigas(fig_Lref_prederrors, fullfile(expmt_dir, strcat(dataout_file, '_prediction_Lref_errors')),...
        'Verbose', true);
        
    savefigas(fig_Lpred_prederrors, fullfile(expmt_dir, strcat(dataout_file, '_prediction_Lpred_errors')),...
        'Verbose', true);
       
    savefigas(fig_Ldiff_prederrors, fullfile(expmt_dir, strcat(dataout_file, '_prediction_Ldiff_errors')),...
        'Verbose', true);
        
end


%% Termination
disp("Press [ENTER] to finish the program");
pause;
close all;
disp("Program Completed.");


%% Helper functions
% compute errors between two shapes of the same size
function errors = compute_errors(shape_ref, shape_pred)
    arguments
        shape_ref (3,:);
        shape_pred (3,:);
    end
        
    % compute distances
    devs = shape_ref - shape_pred;
    dists_sqr = dot(devs, devs, 1);
    
    errors.Min = sqrt(min(dists_sqr(2:end)));
    errors.Max = sqrt(max(dists_sqr(2:end)));
    errors.RMSE = sqrt(mean(dists_sqr(1:end)));
    errors.Tip = sqrt(dists_sqr(end));
    errors.In_Plane = mean(vecnorm(devs([2,3], :), 2,1)); % mean in-plane error
    errors.Out_Plane = mean(vecnorm(devs([1,3], :), 2,1)); % mean out-of-plane error

end

% compute errors between FBG shape and camera shape THERE IS AN ERROR HERE
function [errors, varargout] = compute_camera_errors(shape_cam, shape_fbg, ds)
    arguments
        shape_cam (3,:);
        shape_fbg (3,:);
        ds double = 0.5;
    end
    
    % find each curve arclength
    arclen_cam = arclength(shape_cam');
    arclen_fbg = arclength(shape_fbg');
    
    % generate the arclengths
%     s_cam = 0:ds:arclen_cam;
    s_cam = flip(arclen_cam:-ds:0);
    s_fbg = 0:ds:arclen_fbg;
    
    N = min(numel(s_cam(s_cam>40)), numel(s_fbg(s_fbg > 40)));
    
    % interpolate the shapes
    shape_cam_interp = interp_pts(shape_cam', s_cam);
    shape_fbg_interp = interp_pts(shape_fbg', s_fbg);
    
    % register camera -> FBG
    [R, p] = point_cloud_reg_tip(shape_cam_interp(end-N+1:end,:), ...
                                 shape_fbg_interp(end-N+1:end,:));
    varargout{2} = R; varargout{3} = p;
    
    % transform camera -> FBG
    shape_cam_interp_tf = shape_cam_interp * R' + p';
    varargout{1} = shape_cam_interp_tf';
    
    % compute the errors
    N = min(numel(s_cam), numel(s_fbg));
    errors = compute_errors(shape_cam_interp_tf(end-N+1:end,:)', ...
                            shape_fbg_interp(end-N+1:end,:)');
    
end

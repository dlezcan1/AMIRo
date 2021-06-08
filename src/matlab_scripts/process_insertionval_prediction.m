%% process_insertionval_predition
%
% script to process insertion validation prediction errors
%
% - written by: Dimitri Lezcano

%% Set-up
clear;
configure_env on;
set(0,'DefaultAxesFontSize',20)

% directories to iterate through
expmt_dir = "../../data/needle_3CH_4AA_v2/Insertion_Experiment_04-12-21/";
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

% saving options
save_bool = true;
dataout_file = "FBGdata_predition";
if use_weights
    dataout_file = strcat(dataout_file, "_FBG-weights");
    data_file = "FBGdata_FBG-weights_3d-params.txt";
end

% needle parameters
needleparams = load("../../shape-sensing/shapesensing_needle_properties.mat");

%% Iterate over the trial directories
dir_prev = "";
hole_num_prev = 0;

% generate result table
result_tbl = table('Size', [0,10], 'VariableTypes', ...
        {'uint8', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'},...
        'VariableNames', {'Ins_Hole', 'L_ref', 'L_pred', 'ref_shape', 'pred_shape', 'q_optim', ...
        'RMSE', 'MinError', 'MaxError', 'TipError'});
result_tbl.Properties.VariableUnits = {'', 'mm', 'mm', 'mm', 'mm', 'mm', '', 'mm', 'mm', 'mm'};
for i = 1:length(trial_dirs)   
    % trial operations
    L = str2double(trial_dirs(i).name);
    re_ret = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
    hole_num = str2double(re_ret{1}{1});
    
    % load the needle shape param file
    d = fullfile(trial_dirs(i).folder, trial_dirs(i).name);
    in_file = fullfile(d, data_file);
    [kc, w_init, ~, theta0] = readin_fbgparam_file(in_file);
    
    % generate the reference shape
    [~, p_ref] = fn_intgEP_1layer_Dimitri(kc, w_init, theta0, L, 0, ds, ...
                needleparams.B, needleparams.Binv);
        
    % predict the insertion shapes
    [p_preds, ~, ~, q_best] = predict_insertion_singlelayer(L_preds, L, kc, w_init, needleparams, ...
            p, 'optim_lb', 0.0, 'optim_ub', 1.0, 'theta0', theta0, ...
            'optim_display', 'none');
    
    if ~iscell(p_preds) % cell-ify p_preds
        p_preds = {p_preds};
    end
    
    % compute errors and append to table
    for i = 1:numel(p_preds)
        % append to table        
        if L == L_preds(i)
            result_tbl = [result_tbl;
                          {hole_num, L, L_preds(i), p_ref, p_preds{i}, q_best, ...
                           0, 0, 0, 0}];
        else
            result_tbl = [result_tbl;
                          {hole_num, L, L_preds(i), zeros(3,0), p_preds{i}, q_best, ...
                           0, 0, 0, 0}];
        end
           
    end
    
    % generate the insertion prediction error at the end of the insertion hole
    if hole_num_prev > 0 && hole_num_prev ~= hole_num
        fprintf("Completed Insertion Hole #%d.\n", hole_num_prev);
    end
    
    % update previous hole number
    hole_num_prev = hole_num;
    
end
fprintf("Completed Insertion Hole #%d.\n\n", hole_num_prev);

%% Compute errors and fill out table
hole_nums = reshape(unique(result_tbl.Ins_Hole), 1, []);

% iterate through the insertion hole numbers
for hole_num = hole_nums
    fprintf("Computing errors for insertion hole #%d...\n", hole_num);
    subtbl = result_tbl(result_tbl.Ins_Hole == hole_num, :);
        
    % add in all of the reference shapes
    for L_ref = L_preds
        subtbl_ref = subtbl(subtbl.L_pred == L_ref,:);
        p_ref = subtbl_ref(subtbl_ref.L_ref == L_ref, :).ref_shape{1};

        for j = 1:size(subtbl_ref,1)
            % compute errors
            errors = compute_errors(p_ref, subtbl_ref(j,:).pred_shape{1});

            % update table with errors and ref shape
            subtbl_ref(j,:).ref_shape{1} = p_ref;
            subtbl_ref(j,:).RMSE(1) = errors.RMSE;
            subtbl_ref(j,:).MinError(1) = errors.Min;
            subtbl_ref(j,:).MaxError(1) = errors.Max;
            subtbl_ref(j,:).TipError(1) = errors.Tip;
        end

        % update subtbl with subtbl_ref
        subtbl(subtbl.L_pred == L_ref,:) = subtbl_ref;
    end

    % update result_tbl with subtbl
    result_tbl(result_tbl.Ins_Hole == hole_num,:) = subtbl;
    
end

%% Summarization
result_tbl_nosame = result_tbl(result_tbl.L_ref ~= result_tbl.L_pred,:);
result_summ_Lref = groupsummary(result_tbl_nosame, 'L_ref', {'mean', 'max', 'std'},...
    {'RMSE', 'MinError', 'MaxError', 'TipError'});
result_summ_Lpred = groupsummary(result_tbl_nosame, 'L_pred', {'mean', 'max', 'std'},...
    {'RMSE', 'MinError', 'MaxError', 'TipError'});
result_summ_InsHole = groupsummary(result_tbl_nosame, 'Ins_Hole', {'mean', 'max', 'std'}, ...
    {'RMSE', 'MinError', 'MaxError', 'TipError'});

disp("Summary: Per Reference Insertion Length");
disp(result_summ_Lref);
disp(" ");

disp("Summary: Per Predicted Insertion Length");
disp(result_summ_Lpred);
disp(" ");

disp("Summary: Per Insertion Hole");
disp(result_summ_InsHole);
disp(" ");

if save_bool
    result_tbl_file = fullfile(expmt_dir, strcat(dataout_file, "_results"));
    result_tbl_file_xlsx = strcat(result_tbl_file, ".xlsx");
    result_tbl_file_mat = strcat(result_tbl_file, ".mat");

    % write entire results
    writetable(removevars(result_tbl, {'ref_shape', 'pred_shape'}),...
       result_tbl_file_xlsx, 'WriteVariableNames', true, 'Sheet', 'All Results');

    % write summary results
    writetable(result_summ_Lref, result_tbl_file_xlsx, 'WriteVariableNames', true, ...
        'Sheet', 'Summary Lref');
    writetable(result_summ_Lpred, result_tbl_file_xlsx, 'WriteVariableNames', true, ...
        'Sheet', 'Summary Lpred');
    writetable(result_summ_InsHole, result_tbl_file_xlsx, 'WriteVariableNames', true, ...
        'Sheet', 'Summary Insertion Hole');
    
    % display out
    fprintf("Wrote Results to: %s\n", result_tbl_file_xlsx);
    
    % save table to mat file
    save(result_tbl_file_mat, 'result_tbl', 'result_summ_Lref', ...
        'result_summ_Lpred', 'result_summ_InsHole');
    
    % display out 
    fprintf("Wrote Results to: %s\n", result_tbl_file_mat);
    disp(" ");
end
%% Plotting
fig = figure(1);
set(fig, 'units', 'normalized', 'Position', [0.05,0.05,0.9,0.85])
% iterate over each insertion hole
for hole_num = hole_nums
    subtbl_hole = result_tbl(result_tbl.Ins_Hole == hole_num,:);
    
    for i = 1:numel(L_preds)
        L_i = L_preds(i);

        % grab all of the predicted insertion lengths
        subtbl_hole_Li = subtbl_hole(subtbl_hole.L_pred == L_i,:);
        
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
                    pos = subtbl_hole_Li(j,:).ref_shape{1};
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
        xlabel('z (mm)'); ylabel('x (mm)');
        axis equal; grid on;
        xlabel('z (mm)'); ylabel('x (mm)');
        
        if save_bool
           save_dir = fullfile(expmt_dir, sprintf('Insertion%d/',hole_num));
           figfile_savebase = fullfile(save_dir, sprintf(strcat(dataout_file,...
               '_predicted-shape-comp_Lpred-%.1f'),L_i));
           
           savefig(fig, strcat(figfile_savebase, '.fig'));
           fprintf("Saved figure: %s\n", strcat(figfile_savebase, '.fig'))
           saveas(fig, strcat(figfile_savebase, '.png'));
           fprintf("Saved figure: %s\n", strcat(figfile_savebase, '.png'))
        end
        
        pause(1);
    end
end
%% Saving


%% Termination
close all;
disp("Program Completed.");


%% Helper functions
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

end

%% prepare_prediction_data.m
% script to use combined insertion results to prepare ML method for
%   learning future insertion parameters
%
% written by: Dimitri Lezcano

%% Set-up
% load the data
expmt_dir = fullfile('/Volumes/amiro_needle_data/', '3CH-4AA-0004', ...
                     'Insertion_Experiment_Results'); % CAN CHANGE
result_data_file = fullfile(expmt_dir, 'FBG-Camera-Comp_tip-pcr_FBG-weights_combined-results.mat');
act_result_tbl = load(result_data_file).act_result_tbl;
disp("Data loaded.");

% output file set-up
out_dir = expmt_dir; % CAN CHANGE
out_data_file = fullfile(expmt_dir, 'FBG-Camera-Comp-tip-pcr_FBG-weights_combined-prediction-data');
save_bool = true; % CAN CHANGE

%% Separate out 1 and 2 layer data
% columns to keep
cols_keep = {'Experiment', 'Ins_Hole', 'L_ref', 'w_init_1', 'w_init_2', 'w_init_3',...
             'kc1', 'kc2', 'num_layers', 'singlebend'};

singlelayer_data = act_result_tbl(act_result_tbl.num_layers == 1,cols_keep);
doublelayer_data = act_result_tbl(act_result_tbl.num_layers == 2,cols_keep);

% 1 Layer Data
singlelayer_data.kc2 = []; % remove kc2 - unused
singlelayer_data = renamevars(singlelayer_data, 'kc1', 'kc');
singlelayer_data.layer_num(:) = 1;

% 2 Layer Data
% - separate double-layer data into single-layer
doublelayer_1layer_mask = doublelayer_data.kc2 < 0;
doublelayer_data_1 = doublelayer_data(doublelayer_1layer_mask,:);
doublelayer_data_2 = doublelayer_data(~doublelayer_1layer_mask,:);

% - 1-layer inserted only
doublelayer_data_1.kc2 = [];
doublelayer_data_1 = renamevars(doublelayer_data_1, 'kc1', 'kc');
doublelayer_data_1.num_layers(:) = 1;
doublelayer_data_1.layer_num(:) = 1;

% - 2-layer inserted
doublelayer_data_2.kc1 = doublelayer_data_2.kc2; % move kc2 -> kc1
doublelayer_data_2.layer_num(:) = 2;

doublelayer_data_2_1 = doublelayer_data(~doublelayer_1layer_mask,:);
doublelayer_data_2_1.layer_num(:) = 1;

doublelayer_data_2 = [doublelayer_data_2_1;...
                      doublelayer_data_2];

doublelayer_data_2.kc2 = [];
doublelayer_data_2 = renamevars(doublelayer_data_2, 'kc1', 'kc');

% - combine double layer data's
doublelayer_data = [doublelayer_data_1; doublelayer_data_2];

% combine data
all_data = [singlelayer_data; doublelayer_data];
all_data = sortrows(all_data, {'Experiment', 'Ins_Hole','L_ref'});
all_data = movevars(all_data, {'singlebend', 'num_layers', 'layer_num'}, 'After', 'L_ref');

%% Pair for reference and prediction data
% variable names
original_vars = {'w_init_1', 'w_init_2', 'w_init_3', 'kc'};
ref_vars = {'w_init_ref_1', 'w_init_ref_2', 'w_init_ref_3', 'kc_ref', 'L_ref'};
pred_vars = {'w_init_pred_1', 'w_init_pred_2', 'w_init_pred_3', 'kc_pred', 'L_pred'};

% table set-up
all_data_pred = renamevars(all_data, original_vars, ref_vars(1:length(original_vars)));
all_data_pred.L_pred(:) = all_data_pred.L_ref;
all_data_pred.w_init_pred_1(:) = all_data_pred.w_init_ref_1;
all_data_pred.w_init_pred_2(:) = all_data_pred.w_init_ref_2;
all_data_pred.w_init_pred_3(:) = all_data_pred.w_init_ref_3;
all_data_pred.kc_pred(:) = all_data_pred.kc_ref;

% masks
singlebend_mask = all_data_pred.singlebend;
singlelayer_mask = all_data_pred.layer_num == 1;
doublelayer_mask = all_data_pred.layer_num == 2;

% perform pairing
for expmt = unique(all_data.Experiment)'
    fprintf("Processing %s... ", expmt);
    expmt_mask = strcmp(all_data.Experiment, expmt);
    for ins_hole = unique(all_data{expmt_mask,'Ins_Hole'})'
        ins_mask = all_data.Ins_Hole == ins_hole;

        % handle 1 layer single-bend
        layer1_subtbl = all_data(expmt_mask & ins_mask & singlelayer_mask & singlebend_mask,:);
        for L = unique(layer1_subtbl.L_ref)'
            % determine predicted insertion depths
            L_preds = layer1_subtbl.L_ref(layer1_subtbl.L_ref > L);
            [pred_rows, ~] = find(layer1_subtbl.L_ref == L_preds');
            
            % prepare the new table
            temp_tbl = repmat(layer1_subtbl(layer1_subtbl.L_ref == L,:), length(pred_rows),1);
            temp_tbl(:,pred_vars) = layer1_subtbl(pred_rows, [original_vars, 'L_ref']);

            temp_tbl = renamevars(temp_tbl, original_vars, ref_vars(1:length(original_vars)));

            % add this table to the total prediction data
            all_data_pred = [all_data_pred; temp_tbl];
        end
        
        
        % handle 2 layer single-bend
        layer2_subtbl = all_data(expmt_mask & ins_mask & doublelayer_mask & singlebend_mask,:);
        for L = unique(layer2_subtbl.L_ref)'
            % determine predicted insertion depths
            L_preds = layer2_subtbl.L_ref(layer2_subtbl.L_ref > L);
            [pred_rows, ~] = find(layer2_subtbl.L_ref == L_preds');
            
            % prepare the new table
            temp_tbl = repmat(layer2_subtbl(layer2_subtbl.L_ref == L,:), length(pred_rows),1);
            temp_tbl(:,pred_vars) = layer2_subtbl(pred_rows, [original_vars, 'L_ref']);

            temp_tbl = renamevars(temp_tbl, original_vars, ref_vars(1:length(original_vars)));

            % add this table to the total prediction data
            all_data_pred = [all_data_pred; temp_tbl];
        end


        % handle 1 layer double-bend
        layer1_dbl_subtbl = all_data(expmt_mask & ins_mask & singlelayer_mask & ~singlebend_mask,:);
        for L = unique(layer1_dbl_subtbl.L_ref)'
            % determine predicted insertion depths
            L_preds = layer1_dbl_subtbl.L_ref(layer1_dbl_subtbl.L_ref > L);
            [pred_rows, ~] = find(layer1_dbl_subtbl.L_ref == L_preds');
            
            % prepare the new table
            temp_tbl = repmat(layer1_dbl_subtbl(layer1_dbl_subtbl.L_ref == L,:), length(pred_rows),1);
            temp_tbl(:,pred_vars) = layer1_dbl_subtbl(pred_rows, [original_vars, 'L_ref']);

            temp_tbl = renamevars(temp_tbl, original_vars, ref_vars(1:length(original_vars)));

            % add this table to the total prediction data
            all_data_pred = [all_data_pred; temp_tbl];
        end

    end
    fprintf("Completed.\n")
end
disp(" ");

% table formatting
all_data_pred = sortrows(all_data_pred, {'Experiment', 'Ins_Hole','L_ref', 'L_pred'});

%% Saving
if save_bool
    save(strcat(out_data_file, '.mat'), 'all_data', 'all_data_pred');
    fprintf("Saved table to: %s\n", strcat(out_data_file, '.mat'));

    writetable(all_data, strcat(out_data_file, '.xlsx'));
    fprintf("Saved table to: %s\n", strcat(out_data_file, '.xlsx'));
    
    writetable(all_data_pred, strcat(out_data_file, '_prediction.xlsx'));
    fprintf("Saved table to: %s\n", strcat(out_data_file, '_prediction.xlsx'));

end
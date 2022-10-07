%% summarize_prediction_results.m
%
% This is a function to generate summary plots for the prediction results
%
% - written by: Dimitri Lezcano


%% 
function summarize_prediction_results(...
    prediction_results_file, actual_results_file...
)
    %% Arguments block
    arguments
        prediction_results_file string
        actual_results_file string
    end
    
    %% Load the results table
    act_results_tbl  = load( actual_results_file ).act_result_tbl;
    pred_results_tbl = readtable( prediction_results_file, ...
        'TextType', 'string'...
    );

    %% Generate data masks
    % single-bend single-layer
    act_sb_sl_mask  = ...
        act_results_tbl.singlebend ...
        & act_results_tbl.num_layers == 1 ...
    ;
    pred_sb_sl_mask = ...
        pred_results_tbl.singlebend ...
        & pred_results_tbl.num_layers == 1 ...
    ;

    % single-bend double-layer 
    act_sb_sl_mask  = ...
        act_results_tbl.singlebend ...
        & act_results_tbl.num_layers == 2 ...
    ;
    pred_sb_dl_mask = ...
        pred_results_tbl.singlebend ...
        & pred_results_tbl.num_layers == 2 ...
    ;

    % double-bend single-layer mask
    act_db_sl_mask  = ...
        ~act_results_tbl.singlebend ...
        & act_results_tbl.num_layers == 1 ...
    ;
    pred_db_sl_mask = ...
        ~pred_results_tbl.singlebend ...
        & pred_results_tbl.num_layers == 1 ...
    ;
    
    %% label prediction results per experiment
    % prepare prediction results table
    pred_results_tbl.Experiment_Type = repmat(...
        "",...
        size(pred_results_tbl, 1), 1 ...
    );
    
    % Add experiment types
    pred_results_tbl.Experiment_Type(pred_sb_sl_mask) = "single-bend single-layer";
    pred_results_tbl.Experiment_Type(pred_sb_dl_mask) = "single-bend double-layer";
    pred_results_tbl.Experiment_Type(pred_db_sl_mask) = "double-bend single-layer";
    
    % Generate experiment-correspondences
    pred_results_tbl = join( ...
      pred_results_tbl, ...
      unique(...
        act_results_tbl(:, ["Experiment", "TissueType1", "TissueType2"])...
      ), ...
      "Key", "Experiment" ...
    );

    %% Perform statistics on single-bend single-layer results
    

    
end
    
% load_prediction_results.m
%
% Function to load the prediction results 
%
% - written by: Dimitri Lezcano

function results = load_prediction_results(prediction_file)
    %% Arguments block
    arguments
        prediction_file string
    end
    
    %% Load the prediction file
    pred_tbl = readtable(...
        prediction_file, ...
        "TextType", "String" ...
    );
    pred_tbl = renamevars( pred_tbl, ...
        ["Var1"],...
        ["Index"] ...
    );
    
    
    %% Split the table
    results = struct();
    
    % table headers
    data_vars = [ ...
        "Index", "Experiment", "Ins_Hole", ...
        "L_ref", "L_pred", "singlebend", "num_layers", "layer_num",...
        "w_init_ref_1", "w_init_ref_2", "w_init_ref_3", "kc_ref" ...
    ];
    pred_vars = [ ...
        "w_init_pred_1", "w_init_pred_2", "w_init_pred_3", ...
        "kc_pred" ...
    ];
    pred_true_vars = pred_vars + "_true";
    
    % split the table to results
    results.data             = pred_tbl(:, data_vars);
    results.predictions      = pred_tbl(:, pred_vars);
    results.true_predictions = pred_tbl(:, pred_true_vars);
    
end
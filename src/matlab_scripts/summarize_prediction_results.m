%% summarize_prediction_results.m
%
% This is a function to generate summary plots for the prediction results
%
% - written by: Dimitri Lezcano


%% 
function summarize_prediction_results(...
    prediction_results_file, actual_results_file, expmt_assignments_file, ...
    save_dir...
)
    %% Arguments block
    arguments
        prediction_results_file string
        actual_results_file     string
        expmt_assignments_file  string
        save_dir                string
    end
    
    %% Load the results table
    act_results_tbl  = load( actual_results_file ).act_result_tbl;
    pred_results_tbl = readtable( prediction_results_file, ...
        'TextType', 'string'...
    );
    expmt_assmts_tbl = readtable( expmt_assignments_file, ...
        'TextType', 'string'...
    );
    mkdir(save_dir);
    
    %% TODO: update with needle parameters json file
    s_aa = [10, 30, 65, 100];

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

    % mask for no duplicates
    fwdpred_mask = pred_results_tbl.L_ref < pred_results_tbl.L_pred;
    nodup_mask   = ...
        pred_results_tbl.layer_num == 1 ...
        & pred_results_tbl.L_ref ~= pred_results_tbl.L_pred ...
        & fwdpred_mask ...
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
    
    % Add number of active areas active
    pred_results_tbl.Num_Active_Areas = sum(s_aa < pred_results_tbl.L_ref, 2);
    
    % Generate experiment-correspondences
    pred_results_tbl = join( ...
      pred_results_tbl, ...
      unique(...
        act_results_tbl(:, ["Experiment", "TissueType1", "TissueType2"])...
      ), ...
      "Key", "Experiment" ...
    );
    pred_results_tbl.TissueTypes = string(pred_results_tbl.TissueType1);
    pred_results_tbl.TissueTypes(pred_sb_dl_mask) = strcat( ...
        string(pred_results_tbl.TissueTypes(pred_sb_dl_mask)), ...
        "-", ...
        string(pred_results_tbl.TissueType2(pred_sb_dl_mask)) ...
    );
    
    % Generate training-validation data type correspondences
    pred_results_tbl = join( ...
        pred_results_tbl, ...
        expmt_assmts_tbl ...
    );

    % generate training and testing types
    train_mask = strcmp(pred_results_tbl.DataType, "train");
    test_mask  = strcmp(pred_results_tbl.DataType, "test") ...
               | strcmp(pred_results_tbl.DataType, "val");

    % extra variables
    pred_results_tbl.dL = pred_results_tbl.L_pred - pred_results_tbl.L_ref;

    %% Prepare statistics
    error_varnames = reshape(["FBG_Pred_", "FBG_Cam_", "Pred_Cam_"] + [
            "RMSE"; "MaxError"; "InPlaneError"; "OutPlaneError"
        ], ...
    1, []);
    error_sumfuncs = {'mean', 'max', 'std'};

    %% Perform statistics on single-bend single-layer results
    sb_sl_pred_summ_L = groupsummary( ...
        pred_results_tbl(pred_sb_sl_mask, :), ...
        {'L_ref', 'L_pred'}, ...
        error_sumfuncs, ...
        error_varnames ...
    );

    sb_sl_pred_tissue = groupsummary( ...
        pred_results_tbl(pred_sb_sl_mask, :), ...
        {'TissueType1'}, ...
        error_sumfuncs, ...
        error_varnames ...
    );

    sb_sl_pred_inshole = groupsummary( ...
        pred_results_tbl(pred_sb_sl_mask, :), ...
        {'Ins_Hole'}, ...
        error_sumfuncs, ...
        error_varnames ...
    );

%     %% Perform statistics on single-bend double-layer results
%     sb_dl_pred_summ_L = groupsummary( ...
%         pred_results_tbl(pred_sb_dl_mask & nodup_mask, :), ...
%         {'L_ref', 'L_pred'}, ...
%         error_sumfuncs, ...
%         error_varnames ...
%     );
% 
%     sb_dl_pred_tissue = groupsummary( ...
%         pred_results_tbl(pred_sb_dl_mask & nodup_mask, :), ...
%         {'TissueType1', 'TissueType2'}, ...
%         error_sumfuncs, ...
%         error_varnames ...
%     );
% 
%     sb_dl_pred_inshole = groupsummary( ...
%         pred_results_tbl(pred_sb_dl_mask & nodup_mask, :), ...
%         {'Ins_Hole'}, ...
%         error_sumfuncs, ...
%         error_varnames ...
%     );
% 
%     %% Perform statistics on double-bend single-layer results
%     db_sl_pred_summ_L = groupsummary( ...
%         pred_results_tbl(pred_db_sl_mask & nodup_mask, :), ...
%         {'L_ref', 'L_pred'}, ...
%         error_sumfuncs, ...
%         error_varnames ...
%     );
% 
%     db_sl_pred_tissue = groupsummary( ...
%         pred_results_tbl(pred_db_sl_mask & nodup_mask, :), ...
%         {'TissueType1'}, ...
%         error_sumfuncs, ...
%         error_varnames ...
%     );
% 
%     db_sl_pred_inshole = groupsummary( ...
%         pred_results_tbl(pred_db_sl_mask & nodup_mask, :), ...
%         {'Ins_Hole'}, ...
%         error_sumfuncs, ...
%         error_varnames ...
%     );
    
    %% Plotting Setup
    fig_num = 1;
    ax_pos_adj = [0, 0, 0, -0.05]; % adjust the height of the position
    fig_pos = [ 1, 0.0370, 1.0000, 0.8917 ];
    max_yl = 1.25;

    %% Prepare plotting resutls
    plot_results_inputs = cell(1, 3);
    
    % - single-bend single-layer
    plot_results_inputs{1} = struct();
    plot_results_inputs{1}.mask = pred_sb_sl_mask & nodup_mask;
%     plot_resutls_inputs{1}.train_mask = train_mask;
    plot_results_inputs{1}.file_prefix = "Prediction-Results_singlebend-singlelayer";
    plot_results_inputs{1}.title_insert = "Single-Bend Single-Layer";
    plot_results_inputs{1}.num_layers = 1;
    
    % - single-bend double-layer
    plot_results_inputs{2} = struct();
    plot_results_inputs{2}.mask = pred_sb_dl_mask & nodup_mask;
    plot_results_inputs{2}.file_prefix = "Prediction-Results_singlebend-doublelayer";
    plot_results_inputs{2}.title_insert = "Single-Bend Double-Layer";
    plot_results_inputs{2}.num_layers = 2;
    
    % - single-bend single-layer
    plot_results_inputs{3} = struct();
    plot_results_inputs{3}.mask = pred_db_sl_mask & nodup_mask;
    plot_results_inputs{3}.file_prefix = "Prediction-Results_doublebend-singlelayer";
    plot_results_inputs{3}.title_insert = "Double-Bend Single-Layer";
    plot_results_inputs{3}.num_layers = 1;
    
    %% Plot the results
    for i = 1:numel(plot_results_inputs)
        plot_results(...
            pred_results_tbl,...
            plot_results_inputs{i}.mask, ...
            train_mask, ...
            test_mask, ...
            plot_results_inputs{i}.file_prefix, ...
            plot_results_inputs{i}.title_insert, ...
            save_dir, ...
...             plot_results_inputs{i}.num_layers, ...
            'max_yl', max_yl,...
...             'ax_pos_adj', ax_pos_adj,...
            'fig_pos', fig_pos...
        ); 
        close all;
    end
        

end

%% Helper functions
function plot_results( ...
    tbl, mask, train_mask, test_mask, ...
    file_prefix, title_insert, save_dir, options...
)
    %% Arguments block
    arguments
        tbl          table
        mask         logical
        train_mask   logical
        test_mask    logical
        file_prefix  string
        title_insert string
        save_dir     string
        options.ax_top_pos_adj = [0, -0.075, 0, 0.05];
        options.ax_btm_pos_adj = [0, -0.05,  0, 0.05];
        options.fig_pos = [ 1, 0.0370, 1.0000, 0.8917 ];
        options.fig_num = 1;
        options.max_yl = 1.25;
    end
    %% Unpack arguments
    fig_num = options.fig_num;
    ax_top_pos_adj = options.ax_top_pos_adj;
    ax_btm_pos_adj = options.ax_btm_pos_adj;
    fig_pos = options.fig_pos;
    max_yl = options.max_yl;
    
    %% Set-up Plotting variables
    x_vars_labels_files = [
        "L_ref",       "Insertion Depth"          , " (%s)", "insdepth";
        "dL",          "Insertion Depth Increment", " (%s)", "insdepthinc";
        "TissueTypes", "Tissue Stiffness"         , ""     , "stiff";
    ];
    y_vars_labels_endings = [ 
        "_RMSE",          "RMSE"; 
        "_InPlaneError",  "In-Plane Error";
        "_OutPlaneError", "Out-of-Plane Error";
        "_MaxError",      "Max Error";
        
    ];

    data_comps = [
        "FBG_Pred", "FBG-Reconstructed and Predicted",    "FBG2Pred";
        "Pred_Cam", "Stereo-Reconstructed and Predicted", "Cam2Pred";
    ];
    
    %% Plotting
    for ypi = 1:size(data_comps, 1)
        % unpack data comparison names
        data_comp = data_comps(ypi, 1);
        data_ttl  = title_insert + " " + data_comps(ypi, 2);
        data_fpfx = file_prefix  + "_" + data_comps(ypi, 3);
        
        for xi = 1:size(x_vars_labels_files, 1)
            % set-up variables
            x_var      = x_vars_labels_files(xi, 1);
            x_lbl      = x_vars_labels_files(xi, 2);
            x_lbl_unit = x_vars_labels_files(xi, 3);
            x_fpfx     = x_vars_labels_files(xi, 4);

            y_vars = data_comp + y_vars_labels_endings(:, 1);
            y_lbls = y_vars_labels_endings(:, 2);
            
            % set-up titling and file naming
            data_ttl_x  = data_ttl + " per " + x_lbl;
            data_fpfx_x = data_fpfx + "_"    + x_fpfx;

            % plot
            plot_results_variable( ...
                tbl, ...
                mask, ...
                train_mask, test_mask, ...
                x_var, y_vars, ...
                data_fpfx_x, ...
                data_ttl_x, ...
                save_dir, ...
                'ax_top_pos_adj', options.ax_top_pos_adj, ...
                'ax_btm_pos_adj', options.ax_btm_pos_adj, ...
                'fig_num', fig_num, ...
                'x_label', x_lbl + x_lbl_unit, ...
                'y_labels', y_lbls, ...
                'super_title', data_ttl, ...
                'error_units', 'mm' ...
            );
        
            % increment figure name
            fig_num = fig_num + 1;
        end
    end
end

% plot results for a dynamic variable listing
function plot_results_variable( ...
    tbl, mask, train_mask, test_mask, x_var, y_vars, ...
    file_prefix, title_insert, save_dir, options...
)
    %% Arguments block
    arguments
        tbl          table
        mask         logical
        train_mask   logical
        test_mask    logical
        x_var        string
        y_vars       string
        file_prefix  string
        title_insert string
        save_dir     string
        options.ax_top_pos_adj = [0, -0.075, 0, 0.05];
        options.ax_btm_pos_adj = [0, -0.05,  0, 0.05];
        options.fig_pos = [ 1, 0.0370, 1.0000, 0.8917 ];
        options.fig_num = 1;
        options.max_yl = 1.25;
        options.x_label  = x_var;
        options.y_labels = y_vars;
        options.error_units = "mm";
        options.super_title = ""
    end
    %% Unpack arguments
    fig_num = options.fig_num;
    ax_top_pos_adj = options.ax_top_pos_adj;
    ax_btm_pos_adj = options.ax_btm_pos_adj;
    fig_pos = options.fig_pos;
    max_yl = options.max_yl;
    Ny = numel(y_vars);
    
    %% Function
    fig = figure(fig_num);
    set(fig, 'units', 'normalized', 'position',fig_pos);
    
    axes = [];
    for i = 1:numel(y_vars)
        y_var = y_vars(i);
        % top: training
        axt = subplot(2, Ny, i);
        axt.Position = axt.Position + ax_top_pos_adj;
        boxplot(...
            tbl{mask & train_mask, y_var}, ...
            tbl{mask & train_mask, x_var}  ...
        );
        if i == 1
            ylabel(sprintf("Training Error (%s)", options.error_units));
        end
        title(...
            sprintf(options.y_labels(i), options.error_units),...
            'FontSize', 20 ...
        );
        
        % bottom: validation
        axb = subplot(2, Ny, i+Ny);
        axb.Position = axb.Position + ax_btm_pos_adj;
        boxplot(...
            tbl{mask & test_mask, y_var}, ...
            tbl{mask & test_mask, x_var}  ...
        );
        if i == 1
            ylabel(sprintf("Validation Error (%s)", options.error_units));
        end
        xlabel(sprintf(options.x_label, options.error_units));
        
        % update max ylim
        max_yl = max([axt.YLim, axb.YLim]);
        
        % add axes
        axes = [ axes, axt, axb ];
        
    end
    
    % update ylims
    for i = 1:numel(axes)
        axes(i).YLim = [0, max_yl];
    end
    
    % add super title
    sgtitle(sprintf(...
            "Errors between %s",...
            title_insert   ...
        ),...
        'FontSize', 22, 'FontWeight', 'bold' ...
    );
    
    % saving
    savefigas(...
        fig,  ...
        fullfile(       ...
            save_dir,   ...
            file_prefix ...
        ),    ...
        'Verbose', true ...
    );

end
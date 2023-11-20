%% sensors_update_plots.m

%% Plot Defaults
% variables
linewidth     = 6;
markersize    = 20;
fontsize      = 26;
titlefontsize = round(1.5*fontsize);
fontweight    = 'bold';

% defaults
set(0, 'DefaultLineLineWidth', linewidth);
set(0, 'DefaultAxesFontSize', fontsize);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultTextFontSize', round(fontsize * 1.5));

%% Setup
data_dir = fullfile( ...
    "../../prediction_ml/results" ...
);

model_files = [
    "ff-20_30_40_30_20-ss-shape-mse-transfer-learned-shape-mse-lr-scheduled-0.0001-data-normalized-scaled-length-custom-model_best-train.pt";
    "ff-5_2-shape-mse-kc-activation-sigmoid-winit-activation-tanh-lr-scheduled-0.05-data-normalized-scaled-length-custom-model_best-train.pt"
];

error_file_ext = "_errors.mat";
plot_dir_ext   = ".xlsx-plots";

%% Iterate over the files
for i = 1:numel(model_files)
    %% File & Directory Naming
    full_model_file = fullfile(data_dir, model_files(i));
    full_model_pdir = strcat(full_model_file, plot_dir_ext);

    %% load the data
    l = load(strcat(full_model_file, error_file_ext), 'prediction_result_error_tbl');
    prediction_result_error_tbl = l.prediction_result_error_tbl;


    %% Find the best shape
    L_max = max(prediction_result_error_tbl.L_pred);
    L_ref = 75;

    sb_sl_mask = ( ...
        prediction_result_error_tbl.singlebend_pred ...
        & prediction_result_error_tbl.num_layers_pred == 1 ...
    );

    db_sl_mask = ( ...
        ~prediction_result_error_tbl.singlebend_pred ...
        & prediction_result_error_tbl.num_layers_pred == 1 ...
    );

    sb_dl_mask = ( ...
        prediction_result_error_tbl.singlebend_pred ...
        & prediction_result_error_tbl.num_layers_pred == 2 ...
    );

    % initialization
    cam_shape    = struct();
    pred_shape   = struct();
    cam_shape_tf = struct();
    row_idx      = struct();

    % single-bend single-layer
    pred_result_sbsl = prediction_result_error_tbl(...
        prediction_result_error_tbl.L_pred == L_max ...
        & prediction_result_error_tbl.L_ref == L_ref ...
        & sb_sl_mask, ...
        :...
    );
    [~, idx_best] = min(pred_result_sbsl.Pred_Cam_RMSE);

    cam_shape.sbsl    = pred_result_sbsl.cam_shape_act{idx_best};
    pred_shape.sbsl   = pred_result_sbsl.fbg_shape_pred{idx_best};
    pose_cam_fbg      = pred_result_sbsl.pose_fbg_cam{idx_best};
    cam_shape_tf.sbsl = transformPointsSE3(cam_shape.sbsl, pose_cam_fbg, 1);
    row_idx.sbsl      = pred_result_sbsl.Index(idx_best);

    % double-bend single-layer
    pred_result_dbsl = prediction_result_error_tbl(...
        prediction_result_error_tbl.L_pred == L_max ...
        & prediction_result_error_tbl.L_ref == L_ref ...
        & db_sl_mask, ...
        :...
    );
    [~, idx_best] = min(pred_result_dbsl.Pred_Cam_RMSE);

    cam_shape.dbsl    = pred_result_dbsl.cam_shape_act{idx_best};
    pred_shape.dbsl   = pred_result_dbsl.fbg_shape_pred{idx_best};
    pose_cam_fbg      = pred_result_dbsl.pose_fbg_cam{idx_best};
    cam_shape_tf.dbsl = transformPointsSE3(cam_shape.dbsl, pose_cam_fbg, 1);
    row_idx.dbsl      = pred_result_dbsl.Index(idx_best);

    % single-bend double-layer
    pred_result_sbdl = prediction_result_error_tbl(...
        prediction_result_error_tbl.L_pred == L_max ...
        & prediction_result_error_tbl.L_ref == L_ref ...
        & sb_dl_mask, ...
        :...
    );
    [~, idx_best] = min(pred_result_sbdl.Pred_Cam_RMSE);
    
    cam_shape.sbdl    = pred_result_sbdl.cam_shape_act{idx_best};
    pred_shape.sbdl   = pred_result_sbdl.fbg_shape_pred{idx_best};
    pose_cam_fbg      = pred_result_sbdl.pose_fbg_cam{idx_best};
    cam_shape_tf.sbdl = transformPointsSE3(cam_shape.sbdl, pose_cam_fbg, 1);
    row_idx.sbdl      = pred_result_sbdl.Index(idx_best);

    %% Plot the best shape
    expmt_fields  = fields(pred_shape);
    barrier_pos_z = 65;

    for expmt_idx = 1:numel(expmt_fields)
        expmt_field = expmt_fields{expmt_idx};

        plot_barrier = ~strcmp(expmt_field, 'sbsl');
        double_bend  = startsWith(expmt_field, 'db');

        pred_shape_expmt   = pred_shape.(expmt_field);
        cam_shape_tf_expmt = cam_shape_tf.(expmt_field);

        % 3d
        fig_3d = figure(1);
        set(fig_3d, 'Units', 'normalized', 'position', [0.1, 0.1, 0.8, 0.8]);
        hold off;
        plot3( ...
            pred_shape_expmt(1, :), ...
            pred_shape_expmt(2, :), ...
            pred_shape_expmt(3, :), ...
            'b-'...
        ); hold on;
        plot3( ...
            cam_shape_tf_expmt(1, :), ...
            cam_shape_tf_expmt(2, :), ...
            cam_shape_tf_expmt(3, :), ...
            'g--'...
        ); hold on;
    
        legend('Prediction', 'Ground Truth', 'Location','bestoutside')
        axis equal;
    
        xlabel('x (mm)');
        ylabel('y (mm)');
        zlabel('z (mm)');
        sgtitle(...
            strcat(expmt_field, " | ", strrep(model_files(i), '_', '\_')), ...
            'FontSize', 10,...
            'FontWeight',fontweight...
        );
    
        % 2d
        fig_2d = figure(2);
        set(fig_2d, 'Units', 'normalized', 'position', [0.15, -0.05, 0.8, 0.8]);
        
        subplot(2, 1, 1);
        hold off
        plot(pred_shape_expmt(3, :),   pred_shape_expmt(1, :),   'b-');  hold on;
        plot(cam_shape_tf_expmt(3, :), cam_shape_tf_expmt(1, :), 'g--'); hold on;
        plot_labels = {'Prediction', 'Ground Truth'};
        if plot_barrier
            barrier_name = "Tissue Boundary";
            if double_bend
                barrier_name = "Needle Rotation";
            end
            xline(...
                barrier_pos_z, ...
                'r-.',...
                ... barrier_name, ...
                'LineWidth', linewidth, ...
                'FontSize', fontsize/1.5, ...
                'Fontweight', fontweight...
            );
            plot_labels{end+1} = barrier_name;
        end
    
        ylabel('x (mm)');
        axis equal;
        legend(plot_labels, 'Location','northeast', 'fontsize', fontsize/1.25)
    
    
        subplot(2, 1, 2);
        hold off
        plot(pred_shape_expmt(3, :),   pred_shape_expmt(2, :),   'b-');  hold on;
        plot(cam_shape_tf_expmt(3, :), cam_shape_tf_expmt(2, :), 'g--'); hold on;

        if plot_barrier
            barrier_name = "Tissue Boundary";
            if double_bend
                barrier_name = "Needle Rotation";
            end
            xline(...
                barrier_pos_z, ...
                'r-.',...
                ... barrier_name, ...
                'LineWidth', linewidth, ...
                'FontSize', fontsize/1.5, ...
                'Fontweight', fontweight...
            );
        end
        
        xlabel('z (mm)');
        ylabel('y (mm)');
        axis equal;
        sgtitle(...
            strcat(expmt_field, " | ", strrep(model_files(i), '_', '\_')), ...
            'FontSize', 10,...
            'FontWeight',fontweight...
        );

        % Save the figures
        savefigas( ...
            fig_3d, ...
            fullfile(...
                full_model_pdir, ...
                sprintf("Best_Shape_Plot-3D_experiment-%s_row-%04d", expmt_field, row_idx.(expmt_field))...
            ), ...
            'Verbose', 'true'... 
        )
        savefigas( ...
            fig_2d, ...
            fullfile(...
                full_model_pdir, ...
                sprintf("Best_Shape_Plot-2D_experiment-%s_row-%04d", expmt_field, row_idx.(expmt_field))...
            ), ...
            'Verbose', 'true'... 
        )

    end
    close all;
    
    %% Reformat the box plots
    boxplot_figs   = dir(fullfile(full_model_pdir, 'Prediction-Results_all_Cam2Pred*.fig'));
    bp_linewidth   = 3;
    axlbl_fontsize = 20;

    for bp_fig_idx = 1:numel(boxplot_figs)
        bp_fig_file = fullfile(...
            boxplot_figs(bp_fig_idx).folder, ...
            boxplot_figs(bp_fig_idx).name ...
        );

        if endsWith(bp_fig_file, '-edited.fig')
            continue
        end
        % reformat the figure
        fig = openfig(bp_fig_file);

        %% Set all Linewidths 
        set(findobj(fig, 'type', 'line'), 'LineWidth', bp_linewidth)
        set(findall(fig, 'type', 'axes'), 'FontSize', axlbl_fontsize)
        set(findall(fig, '-property', 'FontSize', '-not', 'type', 'axes'), 'FontSize', axlbl_fontsize);
        
        
        %% reformat x-ticks
        if endsWith(bp_fig_file, '_expmt.fig') || endsWith(bp_fig_file, '_stiff.fig')
            for ax = findall(fig, 'type', 'axes')'
                ax.TickLabelInterpreter = 'tex';
                new_fmt = '-shape\newline';
                
                ax.XAxis.TickLabels = strrep(ax.XAxis.TickLabels, '-shape ', new_fmt );
                ax.XAxis.FontSize = 8;
                ax.Position(4) = 0.39;
            end
        end

        %% Save the figure
        savefigas(...
            fig, ...
            strrep(bp_fig_file, '.fig', '-edited'), ...
            'Verbose', true ...
        )

    end

    
    close all;

end
%% loss_plots_for_ieeesensors.m
% clear; close all;
%% Defaults
linewidth  = 6;
markersize = 20;
fontsize   = 26;

set(0, 'DefaultLineLineWidth', linewidth);
set(0, 'DefaultAxesFontSize', fontsize);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultTextFontSize', round(fontsize * 1.5));

%%
data_dir = fullfile( ...
    "../../prediction_ml", ...
    "logs"...
);

data_file = fullfile(data_dir, "training_results.xlsx");

% sheets
param_loss_sheet = "All training results - Params";
shape_loss_sheet = "torch, custom - joint Shape-MSE";
ss_loss_sheet    = "torch, custom - joint SS Shape";

% variable
number_layers = "x_Layers";
number_params = "x_Params";

trainp_rmse   = "bestTrainRMSE";
valp_rmse     = "bestValRMSE";
trains_rmse   = trainp_rmse + "_mm_";
vals_rmse     = valp_rmse + "_mm_";


%% Load the data sheets
param_loss_tbl = readtable(data_file, 'Sheet', param_loss_sheet);
shape_loss_tbl = readtable(data_file, 'Sheet', shape_loss_sheet, 'MissingRule', 'omitrow', 'TextType', 'string');
ss_loss_tbl    = readtable(data_file, 'Sheet', ss_loss_sheet, 'MissingRule', 'omitrow', 'TextType', 'string');

% points to plot
models_to_plot = [
    2, 7/2, "N1";
    2, 10/2, "N2";
    5, 140/5, "N3";
    5, 120/5, "N4";
];

%% Plot shape_loss
mask_shape_loss = ~strcmp(shape_loss_tbl.WeightInitializer, "Transfer Learning") & shape_loss_tbl.(number_layers) <= 5;
shape_loss_data = shape_loss_tbl(mask_shape_loss, [number_layers, number_params, trains_rmse, vals_rmse] );
shape_loss_name = "Shape RMSE Loss (mm)";

zlimits = [
    min(shape_loss_data{:, [trains_rmse, vals_rmse]}, [], 'all'), ...
    max(shape_loss_data{:, [trains_rmse, vals_rmse]}, [], "all")
];

fig_t = plot_surf( ...
    shape_loss_data.(number_layers), ...
    shape_loss_data.(number_params), ...
    shape_loss_data.(trains_rmse), ...
    models_to_plot, ...
    shape_loss_name, ...
    figure(1)...
);
ax1 = gca();
clim(zlimits);
title(["Supervised Learning Train RMSE Loss", newline]);

fig_v = plot_surf( ...
    shape_loss_data.(number_layers), ...
    shape_loss_data.(number_params), ...
    shape_loss_data.(vals_rmse), ...
    models_to_plot, ...
    shape_loss_name, ...
    figure(2)...
);
ax2 = gca();
clim(zlimits);
title(["Supervised Learning Validation RMSE Loss", newline]);

linkaxes([ax1, ax2])

% plot_surf( ...
%     shape_loss_data.(number_layers), ...
%     shape_loss_data.(number_params), ...
%     abs(shape_loss_data.(trains_rmse) - shape_loss_data.(vals_rmse)), ...
%     models_to_plot, ...
%     figure(5)...
% );
% title(["Supervised Learning Train - Val. Loss", newline])


savefigas(fig_t, fullfile(data_dir, 'shape_train_loss_plot'), 'Verbose', true);
savefigas(fig_v, fullfile(data_dir, 'shape_val_loss_plot'), 'Verbose', true);


%% Plot Self-supervised learning
mask_ss_loss = ss_loss_tbl.(number_layers) <= 5;
ss_loss_data = ss_loss_tbl(mask_ss_loss, [number_layers, number_params, trains_rmse, vals_rmse] );
ss_loss_data.(trains_rmse) = str2double(ss_loss_data.(trains_rmse));
ss_loss_data.(vals_rmse) = str2double(ss_loss_data.(vals_rmse));

ss_loss_name = "SS Shape RMSE Loss (mm)";

zlimits = [
    min(ss_loss_data{:, [trains_rmse, vals_rmse]}, [], 'all')*0.9, ...
    max(ss_loss_data{:, [trains_rmse, vals_rmse]}, [], "all")
];


fig_t = plot_surf( ...
    ss_loss_data.(number_layers), ...
    ss_loss_data.(number_params), ...
    ss_loss_data.(trains_rmse), ...
    models_to_plot, ...
    ss_loss_name, ...
    figure(3)...
);
ax1 = gca();
clim(zlimits);
title(["Self-Supervised Learning Train RMSE Loss", newline]);

fig_v = plot_surf( ...
    ss_loss_data.(number_layers), ...
    ss_loss_data.(number_params), ...
    ss_loss_data.(vals_rmse), ...
    models_to_plot, ...
    ss_loss_name, ...
    figure(4)...
);
ax2 = gca();
clim(zlimits);
title(["Self-Supervised Learning Validation RMSE Loss", newline]);

linkaxes([ax1, ax2])

% plot_surf( ...
%     ss_loss_data.(number_layers), ...
%     ss_loss_data.(number_params), ...
%     abs(ss_loss_data.(trains_rmse) - ss_loss_data.(vals_rmse)), ...
%     models_to_plot, ...
%     figure(6)...
% );
% title(["Self-Supervised Learning Train - Val. Loss", newline])

savefigas(fig_t, fullfile(data_dir, 'ss_train_loss_plot'), 'Verbose', true);
savefigas(fig_v, fullfile(data_dir, 'ss_val_loss_plot'), 'Verbose', true);


%% functions
function fig = plot_surf(num_layers, num_params, error, pts_to_plot, loss_name, fig)
    num_avg_params = num_params ./ num_layers;
    mdl = scatteredInterpolant( ...
        num_layers, ...
        num_avg_params, ...
        error, ...
        'linear', ...
        'nearest'...
    );
    [surfX, surfY] = meshgrid(unique(num_layers),  unique(num_avg_params));
    surfZ = mdl(surfX, surfY);
    
    hold off;
    fig.Units = 'normalized';
    fig.Position(3:4) = [0.6,0.8];
    colormap('jet')

    surf(surfX, surfY, surfZ, 'EdgeColor','none', 'FaceColor','interp'); hold on;
    cb = colorbar();
    cb.Label.String = loss_name;
    cb.Label.Rotation = 270;
    cb.Label.FontSize = 28;
    xlim([2, 5])
    view(0, 90)

    lbl_plot = pts_to_plot(:, 3);
    coords_plot = str2double([pts_to_plot(:, 1), pts_to_plot(:, 2), 100*ones(size(pts_to_plot, 1), 1)]);
    coordst_plot = coords_plot ...
        + [0.05, 0, 0] .* (coords_plot(:, 1) == 2) ...
        - [0.2, 0, 0] .* (coords_plot(:, 1) == 5);
    
    grid_coords = [num_layers, num_avg_params, 100*ones(size(num_layers, 1), 1)];

    plot3(coords_plot(:, 1), coords_plot(:, 2), coords_plot(:,3), 'xm', 'MarkerSize', 20); hold on;
    % plot3(grid_coords(:, 1), grid_coords(:, 2), grid_coords(:,3), 'ok', 'MarkerSize', 20); hold on;
    text(coordst_plot(:, 1), coordst_plot(:, 2), coordst_plot(:, 3), lbl_plot,...
        'color', 'magenta', 'fontsize', 22)
    xticks(unique(surfX));
    axis tight;
    xlabel("Number of Hidden Layers")
    ylabel("Average Size of Hidden Layers"); 
    

end
%% loss_plots_for_ieeesensors.m

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

%% Plot param loss
mask_param_loss = param_loss_tbl.(number_layers) <= 5;
param_loss_data = param_loss_tbl(mask_param_loss, [number_layers, number_params, trainp_rmse, valp_rmse]);

mdl = scatteredInterpolant( ...
    param_loss_data.(number_layers), ...
    param_loss_data.(number_params), ...
    log(param_loss_data.(trainp_rmse)), ...
    'linear', ...
    'nearest'...
);
[surfX, surfY] = meshgrid(unique(param_loss_data.(number_layers)),  unique(param_loss_data.(number_params)));
surfZ = mdl(surfX, surfY);

fig = figure(1);
fig.Units = 'normalized';
fig.Position(3:4) = [0.5,0.8];
colormap('jet')
surf(surfX, surfY, surfZ, 'EdgeColor','none', 'FaceColor','interp');
colorbar()
view(0, 90)
xticks(unique(surfX));
axis tight;
xlabel("Number of Hidden Layers")
ylabel("Number of Network Weights")
title(["Supervised Learning from L2 Loss", newline]);


savefigas(fig, fullfile(data_dir, 'param_loss_plot'), 'Verbose', true);



%% Plot shape_loss
mask_shape_loss = ~strcmp(shape_loss_tbl.WeightInitializer, "Transfer Learning") & shape_loss_tbl.(number_layers) <= 5;
shape_loss_data = shape_loss_tbl(mask_shape_loss, [number_layers, number_params, trains_rmse, vals_rmse] );

mdl = scatteredInterpolant( ...
    shape_loss_data.(number_layers), ...
    shape_loss_data.(number_params), ...
    log(shape_loss_data.(trains_rmse)), ...
    'linear', ...
    'nearest'...
);
[surfX, surfY] = meshgrid(unique(shape_loss_data.(number_layers)),  unique(shape_loss_data.(number_params)));
surfZ = mdl(surfX, surfY);

fig = figure(2);
fig.Units = 'normalized';
fig.Position(3:4) = [0.5,0.8];
colormap('jet')
surf(surfX, surfY, surfZ, 'EdgeColor','none', 'FaceColor','interp');
colorbar()
view(0, 90)
xticks(unique(surfX));
axis tight;
xlabel("Number of Hidden Layers")
ylabel("Number of Network Weights")
title(["Supervised Learning from Shape MSE Loss", newline]);

savefigas(fig, fullfile(data_dir, 'shape_loss_plot'), 'Verbose', true);


%% Plot Self-supervised learning
mask_ss_loss = ss_loss_tbl.(number_layers) <= 5;
ss_loss_data = ss_loss_tbl(mask_ss_loss, [number_layers, number_params, trains_rmse, vals_rmse] );
ss_loss_data.(trains_rmse) = str2double(ss_loss_data.(trains_rmse));
ss_loss_data.(vals_rmse) = str2double(ss_loss_data.(vals_rmse));


mdl = scatteredInterpolant( ...
    ss_loss_data.(number_layers), ...
    ss_loss_data.(number_params), ...
    log(ss_loss_data.(trains_rmse)), ...
    'linear', ...
    'nearest'...
);
[surfX, surfY] = meshgrid(unique(ss_loss_data.(number_layers)),  unique(ss_loss_data.(number_params)));
surfZ = mdl(surfX, surfY);

fig = figure(3);
fig.Units = 'normalized';
fig.Position(3:4) = [0.5,0.8];
colormap('jet')
surf(surfX, surfY, surfZ, 'EdgeColor','none', 'FaceColor','interp');
colorbar()
view(0, 90)
xticks(unique(surfX));
axis tight;
xlabel("Number of Hidden Layers")
ylabel("Number of Network Weights")
title(["Self-Supervised Learning from SS-Shape MSE Loss", newline]);

savefigas(fig, fullfile(data_dir, 'ss_loss_plot'), 'Verbose', true);
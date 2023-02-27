% test_needle_shape_parameters.m
configure_env on;
%% Load parameters
% needle mechanical properties
needle_gauge = 18;
needle_mechparam_file = fullfile('../../shape-sensing', ...
    sprintf('shapesensing_needle_properties_%dG.mat', needle_gauge));
needle_mechparams = load(needle_mechparam_file);

% shape parameters
L  = 200; % mm
ds = 0.5;

%% Get the needle shapes
shape_params = [];

shape_params(1).kc    = 0;
shape_params(1).winit = [0; 0; 0.01];

shape_params(2).kc    = 0;
shape_params(2).winit = [0; 0; 0.02];

shape_params(3).kc    = 0;
shape_params(3).winit = [0.01; 0.01; 0.01];

shape_params(4).kc    = 0.002;
shape_params(4).winit = [0.002; 0.0; 0.0];

shape_params(5).kc    = 0.000282;
shape_params(5).winit = [0.000242; -6.32e-4; -1e-5 ];

shape_params(6).kc    = 5.63e-4;
shape_params(6).winit = [0.0002397; -6.64e-5; -1e-5 ];

%% Generate the needle shapes
for i = 1:numel(shape_params)
    pmat_i = singlebend_singlelayer_shape(...
        shape_params(i).kc,...
        shape_params(i).winit, ...
        L, ...
        needle_mechparams, ...
        0, ...
        ds ...
    );

    shape_params(i).pmat = pmat_i;
end
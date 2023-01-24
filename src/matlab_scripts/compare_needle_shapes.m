%% compare_needle_shapes.m
% this is a function to compare needle shapes
%
% - written by: Dimitri Lezcano

%% Load the standard parameters for the needle
needle_mechparams = load('../../shape-sensing/shapesensing_needle_properties_18G.mat');

% shape parameters
ds = 0.5;
theta0 = 0;

%% Config 1 parameters
winit_1 = [0.002306214	0.000287252	-0.009988025];
kc1_1   = 0.001567341;
L_1     = 125;
name_1  = 'true';

%% Config 2 parameters
winit_2 = [0.002008496	0.000784666	-0.005575825];
kc1_2   = 0.001435876;
L_2     = 125;
name_2  = 'predicted';

%% Get needle shape 1
[pmat_1, Rmat_1, wv_1] = singlebend_singlelayer_shape( ...
    kc1_1, ...
    winit_1, ...
    L_1, ...
    needle_mechparams, ...
    theta0, ...
    ds ...
);

%% Get needle shape 2
[pmat_2, Rmat_2, wv_2] = singlebend_singlelayer_shape( ...
    kc1_2, ...
    winit_2, ...
    L_2, ...
    needle_mechparams, ...
    theta0, ...
    ds ...
);

%% Compare the shapes
errors = compute_shape_errors(pmat_1, pmat_2);
disp(errors);

%% Plot the shapes
figure(1);
subplot(2,1,1);
plot(pmat_1(3,:), pmat_1(1,:), 'DisplayName', name_1); hold on;
plot(pmat_2(3,:), pmat_2(1,:), 'DisplayName', name_2); hold on;
hold off;
ylabel('x [mm]');
axis equal;
grid on;

subplot(2,1,2);
plot(pmat_1(3,:), pmat_1(2,:), 'DisplayName', name_1); hold on;
plot(pmat_2(3,:), pmat_2(2,:), 'DisplayName', name_2); hold on;
hold off;
ylabel('y [mm]');
xlabel('z [mm]');
legend();

axis equal;
grid on;

%% Helper functions
% compute errors between two shapes of the same size
function errors = compute_shape_errors(shape_ref, shape_pred)
    arguments
        shape_ref (3,:);
        shape_pred (3,:);
    end
    
    M = min(size(shape_ref,2), size(shape_pred,2));
        
    % compute distances
    devs = shape_ref(:, 1:M) - shape_pred(:, 1:M);
    dists_sqr = dot(devs, devs, 1);
    
    errors.Min = sqrt(min(dists_sqr(2:end)));
    errors.Max = sqrt(max(dists_sqr(2:end)));
    errors.RMSE = sqrt(mean(dists_sqr(1:end)));
    errors.Tip = sqrt(dists_sqr(end));
    errors.In_Plane = mean(vecnorm(devs([2,3], :), 2,1)); % mean in-plane error
    errors.Out_Plane = mean(vecnorm(devs([1,3], :), 2,1)); % mean out-of-plane error

end
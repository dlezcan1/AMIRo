%% analyze_ros_insertion_experiments_ct
%
% script to analyze the CT-needle insertion results
%
% - written by: Dimitri Lezcano


%% Filename and Data loading
data_dir = fullfile( ...
    ... "/Volumes/dlezcan1-data",...
    "~/data", ...
    "7CH-4AA-0001-MCF-even", ...
    "2023-06-15_2023-06-16_Beef-Insertion-Experiment" ...
);

experiment_result_filename = "all_experiment_results.mat";
analyzed_result_filename   = strrep(experiment_result_filename, ".mat", "_analyzed.mat");

% options (CAN CHANGE)
recalculate_transform_box2needle = false;
recompile_experiment_results     = false;
plot_shapes                      = false;

%% Stack the results tables
if recompile_experiment_results
    disp("Recompiling experiment results...");

    expmt_result_files = dir(fullfile(data_dir, "experiment_results", "*_experiment_results.mat"));
    expmt_result_files = expmt_result_files(~startsWith({expmt_result_files.name}, '._'));

    results_tbl = [];
    for i = 1:numel(expmt_result_files)
        if strcmp(expmt_result_files(i).name, experiment_result_filename)
            continue;
        end
        l = load(fullfile(expmt_result_files(i).folder, expmt_result_files(i).name));

        if isempty(results_tbl)
            results_tbl = l.results_tbl;
            continue
        end
        
        results_tbl = [results_tbl; l.results_tbl];
    end

    results_tbl = sortrows(...
        results_tbl,...
        {'insertion_number', 'insertion_depth'}...
    );
    save(fullfile(data_dir, "experiment_results", experiment_result_filename), 'results_tbl');
    fprintf(...
        "Saved joint experiment resutls to: %s\n",...
        fullfile(data_dir, "experiment_results", experiment_result_filename)...
    );
end

%% Load the Results
l = load(fullfile(data_dir, "experiment_results", experiment_result_filename));
expmt_results = l.results_tbl;

% unique days
expmt_days = unique(dateshift(expmt_results.timestamp, 'start', 'day'));

%% Compute the CT registration
if ~isvariable(expmt_results, 'transform_box2needle') || recalculate_transform_box2needle
    for expmt_day = expmt_days'
        % get the experiments from each day
        mask_day = isbetween(...
            expmt_results.timestamp, ...
            expmt_day, ...
            expmt_day + days(1)...
        );
        mask_depth = expmt_results.insertion_depth > 100;

        expmt_results_day = expmt_results(mask_day & mask_depth, :);

        % transform the CT points using the fiducial_pose (fiducial pose: box -> CT)
        ct_shape_tf = rowfun( ...
            @(shape, pose) {transformPointsSE3(shape{1}, finv(pose{1}), 2)}, ...
            expmt_results_day, ...
            'InputVariables', {'ct_shape', 'fiducial_pose'}, ...
            'OutputVariableNames', {'ct_shape_tf_box'}...
        );

        % interpolate and match each needle and CT points
        ds_interp = 0.5;
        ct_shape_interp_tf = rowfun( ...
            @(shape) {interpolate_shape_uniform(shape{1}, ds_interp, true)}, ...
            ct_shape_tf, ...
            'OutputVariableNames', {'ct_shape_interp_tf_box'}...
        );
        needle_shape_interp = rowfun( ...
            @(shape) {interpolate_shape_uniform(shape{1}, ds_interp, false)}, ...
            expmt_results_day, ...
            'InputVariables', {'needle_shape'},...
            'OutputVariableNames', {'needle_shape_interp'}...
        );
        points_interp = [ct_shape_interp_tf, needle_shape_interp];

        % aggregate points from all the experiments
        points_sampled = rowfun( ...
            @(ct, ndl) match_points(ct{1}, ndl{1}, true), ...
            points_interp, ...
            "NumOutputs", 2, ...
            "InputVariables",{'ct_shape_interp_tf_box', 'needle_shape_interp'}, ...
            "OutputVariableNames", {'ct_shape_interp_sample_tf_box', 'needle_shape_interp_sample'} ...
        );
        agg_points_ct  = cat(1, points_sampled.ct_shape_interp_sample_tf_box{:});
        agg_points_ndl = cat(1, points_sampled.needle_shape_interp_sample{:});

        % compute point cloud registration
        [R_box2needle, ~] = point_cloud_reg(agg_points_ct, agg_points_ndl);
        R_box2needle = round(R_box2needle, 0);
        p_box2needle = mean(agg_points_ndl - agg_points_ct * R_box2needle', 1);
        F_box2needle = makeSE3(R_box2needle, p_box2needle, 'Check', true);

        % add the results to the day
        expmt_results{mask_day, 'transform_box2needle'} = repmat(...
            {F_box2needle}, ...
            sum(mask_day), ...
            1 ...
        );
    end
    results_tbl = expmt_results;
    save( ...
        fullfile(data_dir, "experiment_results", experiment_result_filename), ...
        "results_tbl" ...
    );
    fprintf( ...
        "Saved experiment results with box2needle transform to: %s\n", ...
        fullfile(data_dir, "experiment_results", experiment_result_filename)...
    )
end

%% Display Table
disp("Experiment Results Table:")
disp(expmt_results)

%% Compute errors
warning('off', 'MATLAB:table:RowsAddedNewVars');
for row_i = 1:size(expmt_results, 1)
    % compute transform from CT -> Needle frame
    tf_ct2ndl = ( ...
        expmt_results.transform_box2needle{row_i} ...
        * finv(expmt_results.fiducial_pose{row_i}) ...
    );
    
    % compute the errors with the transform
    errors = compute_reg_errors( ...
        expmt_results.ct_shape{row_i}, ...
        expmt_results.needle_shape{row_i}, ...
        0.5, ...
        tf_ct2ndl...
    );

    % append the errors
    
    expmt_results{row_i, "Min_Error"}      = errors.Min;
    expmt_results{row_i, "Max_Error"}      = errors.Max;
    expmt_results{row_i, "Tip_Error"}      = errors.Tip;
    expmt_results{row_i, "RMSE"}           = errors.RMSE;
    expmt_results{row_i, "In-Plane_Error"}  = errors.In_Plane;
    expmt_results{row_i, "Out-Plane_Error"} = errors.Out_Plane;


    % plot the results
    if plot_shapes
        needle_shape = expmt_results.needle_shape{row_i};
        ct_shape_tf  = transformPointsSE3( ...
            expmt_results.ct_shape{row_i},...
            tf_ct2ndl, ...
            2 ...
        );
            
        fig_shapes = figure(1);
        fig_shapes.Position = [100 100 1000 700];
        ax1 = subplot(2, 1, 1); hold off;
        plot(ax1, needle_shape(:, 3), needle_shape(:, 1), 'linewidth', 4);hold on;
        plot(ax1, ct_shape_tf(:, 3), ct_shape_tf(:, 1), 'linewidth', 4); hold on;
    
        ylabel("x [mm]")
        axis equal;
            
        ax2 = subplot(2, 1, 2); hold off;
        plot(ax2, needle_shape(:, 3), needle_shape(:, 2), 'linewidth', 4); hold on;
        plot(ax2, ct_shape_tf(:, 3), ct_shape_tf(:, 2), 'linewidth', 4); hold on;
        
        xlabel("z [mm]")
        ylabel("y [mm]")
    
        axis equal;
        sgtitle(sprintf(...
            "CT Reconstruction vs. Needle Shape: Insertion %d - Depth %.1f",...
            expmt_results.insertion_number(row_i), ...
            expmt_results.insertion_depth(row_i) ...
        ));
        legend({'MCF', 'CT'}, 'Location','best');
        % linkaxes([ax1, ax2]);
        
        pause(2);
    end
end
warning('on', 'MATLAB:table:RowsAddedNewVars');

disp("Experiment Results w/ errors")
disp(expmt_results)

save(fullfile(data_dir, "experiment_results", analyzed_result_filename));
fprintf(...
    "Saved experiment result table w/ errors to: %s\n",...
    fullfile(data_dir, "experiment_results", analyzed_result_filename)...
)

%% Plot the group statistics
error_columns = expmt_results.Properties.VariableNames( ...
    strcmp(expmt_results.Properties.VariableNames, "RMSE") ...
    | endsWith(expmt_results.Properties.VariableNames, "_Error") ...
);

groupsummary(...
    expmt_results,...
    "insertion_depth",...
    {'min', 'max', 'std', 'mean'}, ...
    error_columns...
);

% plot
fig_error_stats = figure(2);
fig_error_stats.Position = [100, 100, 1200, 800];

tl = tiledlayout('flow');
for plot_i = 1:numel(error_columns)
    nexttile
    
    bp = boxplot(...
        expmt_results.(error_columns{plot_i}), ...
        expmt_results.insertion_depth...
    );
    set(bp, 'LineWidth', 2);
    title(strrep(error_columns{plot_i}, "_", " "))

end
linkaxes(fig_error_stats.Children.Children)
sgtitle(fig_error_stats, "Error Statistics for CT-Reconstruction vs. MCF Needle Shape")

for plot_i = 1:prod(tl.GridSize)
    nexttile(plot_i);
    [col, row] = ind2sub(flip(tl.GridSize), plot_i);

    if col == 1
        ylabel("Error [mm]")
    end

    if row == tl.GridSize(1)
        xlabel("Insertion Depth [mm]")
    end
end

savefigas( ...
    fig_error_stats, ...
    fullfile(data_dir, "experiment_results", "ct_needle_shape_results"), ...
    'Verbose', true ...
)


%% Helper functions
function [shape_interp, s_interp] = interpolate_shape_uniform(shape, ds, reversed)
    arguments
        shape (:, 3)
        ds double
        reversed logical = false
    end

    arclen = arclength(shape);

    s_interp = 0:ds:arclen;
    if reversed
        s_interp = flip(arclen:-ds:0);
    end

    shape_interp = interp_pts(shape, s_interp);

end

function [pts1, pts2] = match_points(shape1, shape2, from_end)
    arguments
        shape1 (:, 3)
        shape2 (:, 3)
        from_end logical = false
    end
    
    N = min(size(shape1, 1), size(shape2, 1));

    if from_end
        pts1 = shape1(end-N+1:end, :);
        pts2 = shape2(end-N+1:end, :);
    else
        pts1 = shape1(1:N, :);
        pts2 = shape2(1:N, :);
    end

    pts1 = {pts1};
    pts2 = {pts2};

end

% compute errors between two shapes of the same size
function errors = compute_shape_errors(shape_ref, shape_pred)
    arguments
        shape_ref (3,:);
        shape_pred (3,:);
    end
        
    % compute distances
    M = min(size(shape_ref, 2), size(shape_pred, 2));
    devs = shape_ref(:, end-M+1:end) - shape_pred(:, end-M+1:end);
    dists_sqr = dot(devs, devs, 1);
    
    errors.Min = sqrt(min(dists_sqr(2:end)));
    errors.Max = sqrt(max(dists_sqr(2:end)));
    errors.RMSE = sqrt(mean(dists_sqr(1:end)));
    errors.Tip = sqrt(dists_sqr(end));
    errors.In_Plane = mean(vecnorm(devs([2,3], :), 2,1)); % mean in-plane error
    errors.Out_Plane = mean(vecnorm(devs([1,3], :), 2,1)); % mean out-of-plane error

end

% compute errors between FBG shape and registation shape 
function [errors, varargout] = compute_reg_errors(shape_cam, shape_fbg, ds, pose_fbg_cam)
    arguments
        shape_cam (:, 3);
        shape_fbg (:, 3);
        ds double = 0.5;
        pose_fbg_cam = nan;
    end
    
    % find each curve arclength
    arclen_cam = arclength(shape_cam);
    arclen_fbg = arclength(shape_fbg);
    
    % generate the arclengths
%     s_cam = 0:ds:arclen_cam;
    s_cam = flip(arclen_cam:-ds:0);
    s_fbg = 0:ds:arclen_fbg;
    
    N = min(numel(s_cam(s_cam>40)), numel(s_fbg(s_fbg > 40)));
    
    % interpolate the shapes
    shape_cam_interp = interp_pts(shape_cam, s_cam);
    shape_fbg_interp = interp_pts(shape_fbg, s_fbg);
    
    % register camera -> FBG
    if all(~isnan(pose_fbg_cam), 'all')
       R = pose_fbg_cam(1:3, 1:3);
       p = pose_fbg_cam(1:3, 4);
       varargout{2} = R; varargout{3} = p;
    else
       [R, p] = point_cloud_reg_tip(shape_cam_interp(end-N+1:end,:), ...
                                    shape_fbg_interp(end-N+1:end,:));
       varargout{2} = R; varargout{3} = p;
    end
    
    % transform camera -> FBG
    shape_cam_interp_tf = shape_cam_interp * R' + p';
    varargout{1} = shape_cam_interp_tf';
    
    % compute the errors
    N = min(numel(s_cam), numel(s_fbg));
    errors = compute_shape_errors(shape_cam_interp_tf(end-N+1:end,:)', ...
                            shape_fbg_interp(end-N+1:end,:)');
    
end

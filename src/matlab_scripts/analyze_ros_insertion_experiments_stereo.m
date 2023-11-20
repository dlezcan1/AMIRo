%% analyze_ros_insertion_experiments_stereo
%
% script to analyze the stereo-needle insertion results
%
% - written by: Dimitri Lezcano

global EXPMT_TIMEZONE
EXPMT_TIMEZONE = "America/New_York";

%% Filename and Data loading
data_dir = fullfile( ...
    "../../data", ...
    "7CH-4AA-0001-MCF-even", ...
    "2023-08-21_Phantom-Insertion-Deflection-Experiment", ...
    "processed_bags", ...
    "out-bevel-direction" ...
);


% options
use_post_proc = true;

% file names
experiment_result_filename = "all_experiment_results.mat";
analyzed_result_filename   = strrep(experiment_result_filename, ".mat", "_analyzed.mat");
robot_pose_filename        = "robot_pose.csv";
robot_data_filename        = "robot_timestamp_ranges.csv";
needle_data_filename       = "needle_data.xlsx";
camera_data_filename       = "left-right_3d-pts.csv";

if use_post_proc
    needle_data_filename = strrep(needle_data_filename, ".xlsx", "_post-proc.xlsx");
end

mkdir(fullfile(data_dir, "experiment_results"));

recompile_experiment_results     = true;
plot_shapes                      = true;

%% Stack the results tables
if recompile_experiment_results
    warning('off', 'MATLAB:table:RowsAddedExistingVars');
    disp("Recompiling experiment results...");
    
    % initialization
    columns_types = {
        'insertion_number', 'double';
        'insertion_depth', 'double';
        'timestamp', 'datetime';
        'needle_shape','cell';
        'camera_shape', 'cell';
        'curvature', 'cell';
        'kappa_c', 'cell';
        'w_init', 'cell';
        'camera_pose', 'cell';
    };
    results_tbl = table( ...
        'Size', [0, size(columns_types, 1)], ...
        'VariableNames', columns_types(:, 1), ...
        'VariableTypes', columns_types(:, 2) ...
    );
    results_tbl.timestamp.TimeZone = EXPMT_TIMEZONE;
    
    % insertion directories
    insertion_dirs = dir(fullfile( ...
        data_dir, ...
        "Insertion*"...
    ));

    % iterate over the insertion directories
    for ins_idx_i = 1:numel(insertion_dirs)
        depth_dirs = dir(fullfile( ...
            insertion_dirs(ins_idx_i).folder, ...
            insertion_dirs(ins_idx_i).name ...
        ));
        depth_dirs = depth_dirs(...
            ~strcmp({depth_dirs.name}, ".") ...
            & ~strcmp({depth_dirs.name}, "..") ...
            & [depth_dirs.isdir] ...
        );
        depth_dirs = depth_dirs( ...
            str2double({depth_dirs.name}) ~= 0 ...
        );

        % load robot data
        robot_data = load_robot_data(fullfile( ...
            insertion_dirs(ins_idx_i).folder, ...
            insertion_dirs(ins_idx_i).name, ...
            robot_data_filename...
        ));

        % iterate over depth directories
        for depth_idx_i = 1:numel(depth_dirs)
            depth_dir_i = fullfile( ...
                depth_dirs(depth_idx_i).folder, ...
                depth_dirs(depth_idx_i).name ...
            );

            % load data
            needle_data = load_needle_data(fullfile(...
                depth_dir_i, ...
                needle_data_filename...
            ));
            camera_data = load_camera_data(fullfile( ...
                depth_dir_i, ...
                camera_data_filename...
            ));
            
            % append results
            results_tbl{end+1, 'needle_shape'}   = {needle_data.shape};
            results_tbl{end, 'curvature'}        = {needle_data.curvature};
            results_tbl{end, 'kappa_c'}          = {needle_data.kappa_c};
            results_tbl{end, 'w_init'}           = {needle_data.w_init};
            results_tbl{end, 'insertion_number'} = needle_data.insertion_number;
            results_tbl{end, 'insertion_depth'}  = needle_data.insertion_depth;

            results_tbl{end, 'camera_shape'} = {camera_data.shape};
            results_tbl{end, 'timestamp'}    = mean([ ...
                camera_data.timestamps.left_image, ...
                camera_data.timestamps.right_image...
            ]);
        
        end
    end
    warning('off', 'MATLAB:table:RowsAddedExistingVars');

    save(fullfile(data_dir, "experiment_results", "all_experiment_results.mat"));
end

%% Load the Results
l = load(fullfile(data_dir, "experiment_results", experiment_result_filename));
expmt_results = l.results_tbl;

% unique days
expmt_days = unique(dateshift(expmt_results.timestamp, 'start', 'day'));


%% Display Table
disp("Experiment Results Table:")
disp(expmt_results)

%% Compute errors
warning('off', 'MATLAB:table:RowsAddedNewVars');
for row_i = 1:size(expmt_results, 1)    
    % compute the errors with the transform
    ds = 0.5;
    [errors, s_compare, R_cam2fbg, p_cam2fbg] = compute_reg_errors( ...
        expmt_results.camera_shape{row_i}, ...
        expmt_results.needle_shape{row_i}, ...
        ds ...
    );
    tf_cam2ndl                       = makeSE3(R_cam2fbg, p_cam2fbg);
    expmt_results.camera_pose{row_i} = {tf_cam2ndl};

    % append the errors
    expmt_results{row_i, "Min_Error"}       = errors.Min;
    expmt_results{row_i, "Max_Error"}       = errors.Max;
    expmt_results{row_i, "Tip_Error"}       = errors.Tip;
    expmt_results{row_i, "RMSE"}            = errors.RMSE;
    expmt_results{row_i, "In-Plane_Error"}  = errors.In_Plane;
    expmt_results{row_i, "Out-Plane_Error"} = errors.Out_Plane;


    % plot the results
    if plot_shapes
        needle_shape     = expmt_results.needle_shape{row_i};
        camera_shape_tf  = transformPointsSE3( ...
            expmt_results.camera_shape{row_i},...
            tf_cam2ndl, ...
            2 ...
        );
            
        fig_shapes = figure(1);
        fig_shapes.Position = [50 50 1500 1000];
        ax1 = subplot(4, 1, 1); hold off;
        plot(ax1, needle_shape(:, 3), needle_shape(:, 1), 'linewidth', 4); hold on;
        plot(ax1, camera_shape_tf(:, 3), camera_shape_tf(:, 1), 'linewidth', 4); hold on;
        yline(needle_shape(1, 1), 'k--', 'linewidth', 3)
    
        ylabel("x [mm]")
        axis equal;
            
        ax2 = subplot(4, 1, 2); hold off;
        plot(ax2, needle_shape(:, 3), needle_shape(:, 2), 'linewidth', 4); hold on;
        plot(ax2, camera_shape_tf(:, 3), camera_shape_tf(:, 2), 'linewidth', 4); hold on;
        yline(needle_shape(1, 2), 'k--', 'linewidth', 3)
        
        legend({'MCF', 'Stereo'}, 'Location','best');
        axis equal;
        xlabel("z [mm]")
        ylabel("y [mm]")

        ax3 = subplot(4, 1, 3); hold off;
        plot(ax3, s_compare, errors.deviations(1, :), 'LineWidth',4); hold on;
        plot(ax3, s_compare, errors.deviations(2, :), 'LineWidth',4); hold on;
        plot(ax3, s_compare, errors.deviations(3, :), 'LineWidth',4); hold on;
        yline(0, 'k--', 'linewidth', 3)
        
        title("Error plots")
        legend({"X error", "Y error", "Z error"});
        
        xlabel("Arclength [mm]")
        ylabel("Deviation [mm]")

        ax4 = subplot(4, 1, 4); hold off;
        curvatures = expmt_results.curvature{row_i};
        AA_list = 1:size(curvatures, 2);
        plot(ax4, AA_list, curvatures(1, :), '.', 'MarkerSize', 24); hold on;
        plot(ax4, AA_list, curvatures(2, :), '.', 'MarkerSize', 24); hold on;
        yline(0, 'k--', 'linewidth', 3)

        legend({"X", "Y"});
        ylabel("Curvature [1/m]")
        xlabel("Active Area #")
    
        
        sgtitle(sprintf(...
            "Stereo Reconstruction vs. Needle Shape: Insertion %d - Depth %.1f",...
            expmt_results.insertion_number(row_i), ...
            expmt_results.insertion_depth(row_i) ...
        ));
        % linkaxes([ax1, ax2]);
        
        % saving
        odir = fullfile( ...
            data_dir, ...
            "experiment_results", ...
            "figures", ...
            sprintf("Insertion%d", expmt_results.insertion_number(row_i)), ...
            string(expmt_results.insertion_depth(row_i)) ...
        );
        warning("off", "MATLAB:MKDIR:DirectoryExists")
        mkdir(odir);
        warning("on", "MATLAB:MKDIR:DirectoryExists")
        savefigas(...
            fig_shapes,...
            fullfile(odir, "shape_camera_comparison"), ...
            'Verbose', true ...
        )

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

% remove tip and min error (no need based on registration method)
error_columns = error_columns( ...
    ~strcmp(error_columns, "Min_Error") ...
    & ~strcmp(error_columns, "Tip_Error") ...
);
error_columns = sort(error_columns);
error_columns = {'In-Plane_Error', 'Out-Plane_Error', 'RMSE', 'Max_Error'};

groupsummary(...
    expmt_results,...
    "insertion_depth",...
    {'min', 'max', 'std', 'mean'}, ...
    error_columns...
);

% plot
fig_error_stats = figure(2);
fig_error_stats.Position = [100, 100, 900, 600];

tl = tiledlayout('flow');
for plot_i = 1:numel(error_columns)
    nexttile

    bp_categories = {string(dateshift(expmt_results.timestamp, 'start', 'day')), expmt_results.insertion_depth };
    if numel(expmt_days) == 1
        bp_categories =  {expmt_results.insertion_depth};
    end
    bp = boxplot(...
        expmt_results.(error_columns{plot_i}), ...
        bp_categories ...
    );
    set(bp, 'LineWidth', 2);
    title(strrep(error_columns{plot_i}, "_", " "))

end
linkaxes(fig_error_stats.Children.Children)
sgtitle(fig_error_stats, "Error Statistics for Stereo-Reconstruction vs. MCF Needle Shape")

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

odir = fullfile(data_dir, "experiment_results", "figures");
warning("off", "MATLAB:MKDIR:DirectoryExists")
mkdir(odir);
warning("on", "MATLAB:MKDIR:DirectoryExists")
savefigas( ...
    fig_error_stats, ...
    fullfile(odir, "camera_needle_shape_results"), ...
    'Verbose', true ...
)

%% Plot all needle shapes in separate figures
fig_ndl_shapes = figure(4); hold off;
fig_ndl_shapes.Position = [100, 100, 1600, 900];

fig_cam_shapes = figure(5); hold off;
fig_cam_shapes.Position = [150, 100, 1600, 900];

for expmt_day = expmt_days'
    expmt_day_mask = isbetween(...
        expmt_results.timestamp,...
        expmt_day, ...
        expmt_day + days(1) ...
    );

    ndl_shapes = cat(1, expmt_results.needle_shape{expmt_day_mask});
    cam_shapes  = cat(1, expmt_results.camera_shape{expmt_day_mask});
    
    % plot needle shapes
    figure(fig_ndl_shapes);
    plot3(...
        ndl_shapes(:, 1), ndl_shapes(:, 2), ndl_shapes(:, 3), ...
        '.', 'MarkerSize',24 ...
    ); hold on;

    % Plot stereo shapes
    figure(fig_cam_shapes);
    plot3(...
        cam_shapes(:, 1), cam_shapes(:, 2), cam_shapes(:, 3), ...
        '.', 'MarkerSize',24 ...
    ); hold on;

end
    
figure(fig_ndl_shapes);
legend(string(expmt_days));
xlabel("x [mm]");
ylabel("y [mm]");
zlabel("z [mm]");
title("MCF-Sensed Needle Shapes");
% axis equal; 
grid on;

figure(fig_cam_shapes);
legend(string(expmt_days));   
xlabel("x [mm]");
ylabel("y [mm]");
zlabel("z [mm]");
title("Stereo-Reconstructed Needle Shapes");
% axis equal; 
grid on;

% saving
savefigas( ...
    fig_ndl_shapes, ...
    fullfile(data_dir, "experiment_results", "figures", "aggregated_mcf-needle_shapes"), ...
    "Verbose", true...
);
savefigas( ...
    fig_cam_shapes, ...
    fullfile(data_dir, "experiment_results", "figures", "aggregated_cam-needle_shapes"), ...
    "Verbose", true...
);


%% Helper functions
% load the needle data
function data = load_needle_data(filename)
    global EXPMT_TIMEZONE

    data.shape     = readmatrix(filename, 'Sheet', 'shape');
    data.curvature = reshape(readmatrix(filename, 'Sheet', 'curvature'), 2, []);
    data.kappa_c   = readmatrix(filename, 'Sheet', 'kappa_c');
    data.w_init    = readmatrix(filename, 'Sheet', 'winit');

    % handle timestamps
    filename_original = strrep( ...
        filename, ...
        "_post-proc.xlsx", ...
        ".xlsx" ...
    );
    timestamps = readtable( ...
        filename_original, ...
        "Sheet", "ROS timestamps", ...
        "ReadRowNames", true ...
    );
    timestamps = rowfun( ...
        @(ts) datetime( ...
            ts, ...
            'convertfrom', 'EpochTime', ...
            'tickspersecond', 1e9, ...
            'TimeZone', 'UTC' ...
        ), ...
        timestamps ...
    );
    timestamps = rowfun( ...
        @(ts) setfield(ts, 'TimeZone', EXPMT_TIMEZONE), ...
        timestamps ...
    ); % Convert UTC -> EST
    timestamps = rows2vars(timestamps);
    data.timestamps = table2struct(timestamps(:, 2:end));
    
    % handle insertion information
    filename_split         = pathsplit(filename);
    insertion_number_regex = regexp( ...
        filename_split{end-2}, ...
        "Insertion(\d+)", ...
        'tokens'...
    );
    data.insertion_number  = str2num(insertion_number_regex{1}{1});
    data.insertion_depth   = str2double(filename_split{end-1});

end

function data = load_robot_data(filename)
    global EXPMT_TIMEZONE

    tbl_varname_warnid = 'MATLAB:table:ModifiedAndSavedVarnames';

    warning('off', tbl_varname_warnid);
    data = readtable(...
        filename, ...
        'VariableNamingRule', 'modify', ...
        'ReadRowNames', false, ...
        'ReadVariableNames', true ...
    );
    warning('on', tbl_varname_warnid);
    
    data = renamevars(...
        data, ...
        {'InsertionDepth_mm_'}, ...
        {'InsertionDepth'}...
    );
    
    % convert to datetimes
    data.ts_min = datetime( ...
        data.ts_min, ...
        'convertfrom', 'EpochTime', ...
        'tickspersecond', 1e9, ...
        'TimeZone', 'UTC' ...
    );
    data.ts_max = datetime( ...
        data.ts_max, ...
        'convertfrom', 'EpochTime', ...
        'tickspersecond', 1e9, ...
        'TimeZone', 'UTC' ...
    );

    % timezone change
    data.ts_min.TimeZone = EXPMT_TIMEZONE;
    data.ts_max.TimeZone = EXPMT_TIMEZONE;


end

function data = load_camera_data(filename)
    global EXPMT_TIMEZONE
    
    data.shape = readmatrix(filename);
    
    % handle ROS camera information
    psplit          = pathsplit(filename);
    d               = pathjoin(psplit(1:end-1));
    ros_camera_data = fullfile(d, "camera_data.xlsx");

    if isfile(ros_camera_data)
        timestamps = readtable( ...
            ros_camera_data, ...
            'Sheet', 'ROS timestamps', ...
            'ReadRowNames', true ...
        );
        timestamps = rowfun( ...
            @(ts) datetime( ...
                ts, ...
                'convertfrom', 'EpochTime', ...
                'tickspersecond', 1e9, ...
                'TimeZone', 'UTC' ...
            ), ...
            timestamps ...
        );
        timestamps = rowfun( ...
            @(ts) setfield(ts, 'TimeZone', EXPMT_TIMEZONE), ...
            timestamps ...
        ); % Convert UTC -> EST

        timestamps      = rows2vars(timestamps);
        data.timestamps = table2struct(timestamps(:, 2:end));

    end
end


function [shape_interp, s_interp] = interpolate_shape_uniform(shape, ds, reversed, max_s, min_s)
    arguments
        shape (:, 3)
        ds double
        reversed logical = false
        max_s double = nan;
        min_s double = nan;
    end

    arclen = arclength(shape);

    s_interp = 0:ds:arclen;
    if reversed
        s_interp = flip(arclen:-ds:0);
    end
    if ~isnan(max_s)
        s_interp = s_interp(s_interp <= max_s);
    end


    if ~isnan(min_s) 
        if min_s < 0
            min_s = max(s_interp) + min_s;
        end
        s_interp = s_interp(min_s <= s_interp);
    end
    

    shape_interp = interp_pts(shape, s_interp);

end

function [pts1, pts2] = match_points(shape1, shape2, from_end, idx_min, idx_max)
    arguments
        shape1 (:, 3)
        shape2 (:, 3)
        from_end logical = false
        idx_min = 1
        idx_max = inf
    end
    
    N = min(size(shape1, 1), size(shape2, 1));

    idx_lo = max(1, idx_min);    
    idx_hi = min(N, idx_max);
    if from_end
        pts1 = shape1(end-N+1:end, :);
        pts1 = pts1(idx_lo:idx_hi, :);

        pts2 = shape2(end-N+1:end, :);
        pts2 = pts2(idx_lo:idx_hi, :);
    else
        
        pts1 = shape1(idx_lo:idx_hi, :);
        pts2 = shape2(idx_lo:idx_hi, :);
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
    
    errors.deviations = devs;
    errors.l2_dist = sqrt(dists_sqr);
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
    s_cam = flip(arclen_cam:-ds:0);
    s_fbg = 0:ds:arclen_fbg;
    
    % interpolate the shapes
    shape_cam_interp = interp_pts(shape_cam, s_cam);
    shape_fbg_interp = interp_pts(shape_fbg, s_fbg);
   
    
    % register camera -> FBG
    if all(~isnan(pose_fbg_cam), 'all')
       R = pose_fbg_cam(1:3, 1:3);
       p = pose_fbg_cam(1:3, 4);
       varargout{2} = R; varargout{3} = p;
    else
       N = min(size(shape_cam_interp, 1), size(shape_fbg_interp, 1));
       [R, p] = point_cloud_reg_tip(shape_cam_interp(end-N+1:end,:), ...
                                    shape_fbg_interp(end-N+1:end,:));
       varargout{2} = R; varargout{3} = p;
    end
    
    % transform camera -> FBG
    shape_cam_interp_tf = shape_cam_interp * R' + p';
    
    % % use z-axis interpolation (BAD IDEA)
    % shape_cam_tf        = transformPointsSE3(shape_cam, R, p, 2);
    % shape_cam_interp_tf = interp_pts_col(shape_cam_tf, 2, shape_fbg(:, 2));
    % shape_fbg_interp    = interp_pts_col(shape_fbg, 2, shape_fbg(:, 2));

    % compute the errors
    N = min(numel(s_cam), numel(s_fbg));
    varargout{1} = s_fbg(end-N+1:end);
    errors = compute_shape_errors(shape_cam_interp_tf(end-N+1:end,:)', ...
                            shape_fbg_interp(end-N+1:end,:)');

    % VARIABLE ARGUMENT OUTS
    varargout{4} = shape_cam_interp_tf';
    
end

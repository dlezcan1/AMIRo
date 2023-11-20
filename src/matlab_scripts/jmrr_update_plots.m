% update plots for jmrr
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
jmrr_dir = fullfile( ...
    "/mnt/data-drive/onedrive", ...
    "/Publications/2023 JMRR"...
);

result_mats = dir(fullfile(jmrr_dir, "data", "*experiment-results.mat"));

%% Iterate over results
for i = 1:numel(result_mats)
    l = load( ...
        fullfile(result_mats(i).folder, result_mats(i).name), ...
        'expmt_results'...
    );
    expmt_results = l.expmt_results;
    
    % reformat table
    old_vars = [];
    new_vars = [];
    if isvariable(expmt_results, "insertion_depth")

    elseif isvariable(expmt_results, "L_ref") 
        old_vars = [old_vars, "L_ref"];
        new_vars = [new_vars, "insertion_depth"];
    end
    if isvariable(expmt_results, "fbg_shape")
        old_vars = [old_vars, "fbg_shape"];
        new_vars = [new_vars, "needle_shape"];
    end
    if isvariable(expmt_results, "cam_shape")
        old_vars = [old_vars, "cam_shape"];
        new_vars = [new_vars, "gt_shape"];
    end
    if isvariable(expmt_results, "camera_shape")
        old_vars = [old_vars, "camera_shape"];
        new_vars = [new_vars, "gt_shape"];
    end
    if isvariable(expmt_results, "ct_shape")
        old_vars = [old_vars, "ct_shape"];
        new_vars = [new_vars, "gt_shape"];
        pose_nc = rowfun( ...
            @(fid_pose, tf_box2ndl) {tf_box2ndl{1} * finv(fid_pose{1})}, ...
            expmt_results, ...
            "InputVariables", {'fiducial_pose', 'transform_box2needle'}, ...
            "OutputVariableNames",{'Pose_nc'}...
        );
        expmt_results.Pose_nc = pose_nc.Pose_nc;
    end
    if isvariable(expmt_results, "camera_pose")
        old_vars = [old_vars, "camera_pose"];
        new_vars = [new_vars, "Pose_nc"];
        camera_pose = rowfun( ...
            @(pose) pose{1},expmt_results, ...
            "InputVariables","camera_pose", ...
            "OutputVariableNames","camera_pose"...
        );
        expmt_results.camera_pose = camera_pose.camera_pose;
    end
    if isvariable(expmt_results, "Max_Error") && ~isvariable(expmt_results, "MaxError")
        old_vars = [old_vars, "Max_Error"];
        new_vars = [new_vars, "MaxError"];
    end


    if ~isempty(old_vars) && ~isempty(new_vars)
        expmt_results = renamevars(expmt_results, old_vars, new_vars);
    end
    
    % find best one to plot
    L_max = max(expmt_results.insertion_depth);
    expmt_results_subL = expmt_results(expmt_results.insertion_depth == L_max, :);
    [~, best_idx] = min(expmt_results_subL.RMSE);

    gt_shape_lengths = rowfun( ...
        @(shape) arclength(shape{1}), ...
        expmt_results_subL, ...
        "InputVariables","gt_shape"...
    );

    ndl_shape = expmt_results_subL.needle_shape{best_idx};
    gt_shape  = expmt_results_subL.gt_shape{best_idx};
    pose_nc   = expmt_results_subL.Pose_nc{best_idx};

    gt_shape_tf = transformPointsSE3(gt_shape, pose_nc, 2);

    %% Plot
    fig = figure(1);
    subplot(2, 1, 1); hold off;
    plot( ...
        ndl_shape(:, 3), ...
        ndl_shape(:, 1), ...
        'b-'...
    ); hold on;
    plot( ...
        gt_shape_tf(:, 3), ...
        gt_shape_tf(:, 1), ...
        'r--'...
    ); hold on;
    ylabel("x (mm)");
    legend("Shape-Sensing", "Ground Truth", 'location', 'best')
    axis equal;

    subplot(2, 1, 2); hold off;
    plot( ...
        ndl_shape(:, 3), ...
        ndl_shape(:, 2), ...
        'b-'...
    ); hold on;
    plot( ...
        gt_shape_tf(:, 3), ...
        gt_shape_tf(:, 2), ...
        'r--' ...
    ); hold on;
    xlabel("z (mm)");
    ylabel("y (mm)");
    axis equal;

    sgtitle(strrep(result_mats(i).name, '_', '\_'));
    
    savefigas( ...
        fig, ...
        fullfile("~", strrep(result_mats(i).name, '.mat', '')), ...
        'Verbose', true ...
    );

end

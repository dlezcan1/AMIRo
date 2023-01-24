function rerun_results_tbl(results_tbl_file)
   %% arguments block
    arguments
        results_tbl_file string
    end
    
    %% Setup file names
    out_results_tbl_file  = strrep(results_tbl_file, ".mat", "_recomputed.mat");
    results_tbl_file_path = pathsplit(results_tbl_file);
    
    data_dir = pathjoin(results_tbl_file_path(1:end-2));
    
    %% Load shape parameters
    % shape parameters
    ds             = 0.5;
    theta0         = 0;
    s_crit_dblbend = 65;
    
    % needle mechanical properties
    needle_gauge = 18;
    needle_mechparam_file = fullfile('../../shape-sensing', ...
        sprintf('shapesensing_needle_properties_%dG.mat', needle_gauge));
    needle_mechparams = load(needle_mechparam_file);
    
    
    %% Load the results
    results = load(results_tbl_file);
    act_result_tbl = results.act_result_tbl;
    
    %% Go through all lines and re-write each line
    for i = 1:size(act_result_tbl, 1)
       % unpack parameters
       L     = act_result_tbl{i, 'L_ref'};
       kc1   = act_result_tbl{i, 'kc1'};
       kc2   = act_result_tbl{i, 'kc2'};
       winit = [
           act_result_tbl{i, 'w_init_1'};
           act_result_tbl{i, 'w_init_2'};
           act_result_tbl{i, 'w_init_3'};
       ];
       singlebend = act_result_tbl{i, 'singlebend'};
       num_layers = act_result_tbl{i, 'num_layers'};
       pmat_cam   = act_result_tbl.cam_shape{i};
       
       % - experiment information
       experiment = act_result_tbl.Experiment(i);
       experiment_datadir = get_experiment_path( data_dir, experiment );
       
       % load experiment file
       experiment_info = jsondecode(fileread(fullfile( ...
           experiment_datadir, ...
           "experiment.json" ...
       )));
        
       % Recompute FBG needle shape
       if ~singlebend && L > s_crit_dblbend
           pmat_fbg = doublebend_singlelayer_shape(...
               kc1, ...
               winit, ...
               L, ...
               s_crit_dblbend, ...
               needle_mechparams, ...
               0, ...
               ds ... 
           );
%            continue
           
       elseif num_layers == 2 && kc2 > 0
           pmat_fbg = singlebend_doublelayer_shape(...
               kc1, ...
               kc2, ...
               winit, ...
               L, ...
               experiment_info.tissue1Length, ...
               needle_mechparams, ...
               0, ...
               ds ...
           );
           
%            continue
           
       else %if num_layers == 1
           pmat_fbg = singlebend_singlelayer_shape( ...
               kc1, ...
               winit, ...
               L, ...
               needle_mechparams,... 
               0, ...
               ds ...
           );
       end
       
       % recompute Pose_nc
       [R_nc, p_nc, ...
        cam_pos_interp, s_cam,...
        fbg_pos_interp, s_fbg,...
        cam_pos_interp_tf, fbg_pos_interp_tf] = pcr_Cam2FBG(pmat_cam, pmat_fbg', ds, -1);
       pose_nc = makeSE3(R_nc, p_nc);
       
       % recompute errors
       N_overlap = min(size(cam_pos_interp_tf,1), size(fbg_pos_interp,1));
       errors = error_analysis(cam_pos_interp_tf(end-N_overlap+1:end,:),...
                            fbg_pos_interp(end-N_overlap+1:end,:));
       
       % alter new table values
       act_result_tbl.fbg_shape{i} = pmat_fbg';
       act_result_tbl.Pose_nc{i}   = pose_nc;
       
       act_result_tbl.RMSE(i)     = errors.RMSE;
       act_result_tbl.MaxError(i) = max(errors.L2);
       act_result_tbl.InPlane(i)  = mean(errors.in_plane);
       act_result_tbl.OutPlane(i) = mean(errors.out_plane);       
        
    end
    
    %% Save the new table
    save(out_results_tbl_file, 'act_result_tbl');
    fprintf("Saved re-computed results to: %s\n", out_results_tbl_file);
    
    
    
end

%% Helper functions
function [R_nc, p_nc, varargout] = pcr_Cam2FBG(p_cam, p_fbg, ds, min_s)
    arguments
        p_cam (:,3);
        p_fbg (:,3);
        ds {mustBePositive} = 0.5;
        min_s double = -1;
    end
    
    % interpolate the points
    [p_cam_interp, s_cam_interp] = interpolate_shape(p_cam, ds, true);
    [p_fbg_interp, s_fbg_interp] = interpolate_shape(p_fbg, ds, true);
    
    % ensure sorted arclengths
    [s_cam_interp, cam_idxs] = sort(s_cam_interp);
    p_cam_interp = p_cam_interp(cam_idxs, :);
    [s_fbg_interp, fbg_idxs] = sort(s_fbg_interp);
    p_fbg_interp = p_fbg_interp(fbg_idxs, :);
    
    % Determine the matching points
    N_match = min(sum(s_fbg_interp > min_s), sum(s_cam_interp > min_s));
    
    [R_nc, p_nc] = point_cloud_reg_tip( p_cam_interp(end - N_match + 1:end,:),...
                                        p_fbg_interp(end - N_match + 1:end,:));
    F_nc = makeSE3(R_nc, p_nc);
                                    
    % variable output arguments
    % - first interpolated
    varargout{1} = p_cam_interp;
    varargout{2} = s_cam_interp;
    varargout{3} = p_fbg_interp;
    varargout{4} = s_fbg_interp;
    % - Camera needle shape in needle frame
    varargout{5} = transformPointsSE3(p_cam_interp, F_nc, 2); 
    % - FBG needle shape in camera frame
    varargout{6} = transformPointsSE3(p_fbg_interp, finv(F_nc), 2); 
    
end

% error analysis
function errors = error_analysis(cam, fbg)
% measures error metrics from each points
    
    % L2 distance
    errors.L2 = vecnorm(cam - fbg, 2, 2); 
    
    % component-wise error
    errors.dx = abs(cam(:,1) - fbg(:,1));
    errors.dy = abs(cam(:,2) - fbg(:,2));
    errors.dz = abs(cam(:,3) - fbg(:,3));
    
    % in/out-plane error (assume in-plane is yz and out-plane is xz)
    errors.in_plane = vecnorm(cam(:, 2:3) - fbg(:, 2:3), 2, 2);
    errors.out_plane = vecnorm(cam(:,[1, 3]) - fbg(:, [1,3]), 2, 2);
    
    errors.RMSE = sqrt(mean(errors.L2.^2));
    
end

% interpolate the shape to constant arclength
function [shape_interp, s_interp] = interpolate_shape(shape, ds, flip_s)
    arguments
        shape (:,3);
        ds {mustBePositive} = 0.5;
        flip_s logical = false; % flip the arclength generation from the needle shapes
    end
    
    % determine arclengths
    arclen = arclength(shape);
    
    % generate interpolation arclengths
    if flip_s
        s_interp = flip(arclen:-ds:0);
    else
        s_interp = 0:ds:arclen;
    end
    
    % interpolate the needle shapes
    shape_interp = interp_pts(shape, s_interp);
    
    
end
% evaluate_prediction_results.m
%
% This is a script to evaluate results from prediction data
%
% - written by: Dimitri Lezcano

configure_env on;
clear; close all;
%% Load the data and shape-sensing results
data_dir = "D:/git/amiro/prediction_ml/results";
prediction_files = dir(fullfile(data_dir, "*.xlsx"));
mask_exclude = ...
    endsWith({prediction_files.name}, "_errors.xlsx") ...
;
prediction_files = prediction_files(~mask_exclude);

% actual results
needle_dir = fullfile("../../data", "3CH-4AA-0004");
expmt_results = load( fullfile( ...
        needle_dir,...
        "Insertion_Experiment_Results",...
        "FBG-Camera-Comp_tip-pcr_FBG-weights_combined-results-complete.mat" ...
    ) ...
);
expmt_results_tbl = expmt_results.act_result_tbl;

% needle mechanical properties
needle_gauge = 18;
needle_mechparam_file = fullfile('../../shape-sensing', ...
    sprintf('shapesensing_needle_properties_%dG.mat', needle_gauge));
needle_mechparams = load(needle_mechparam_file);

%% Run prediction results
for i = 1:length(prediction_files)
    %% Load the prediction results
    prediction_file = fullfile( ...
        prediction_files(i).folder, ...
        prediction_files(i).name ...
    );
    prediction_results = load_prediction_results(prediction_file);
    
    fprintf("Analyzing results for predictions in: %s\n", prediction_file);
    
    %% Initialize results table
    results_tbl_varnames = reshape(["FBG_Pred_", "FBG_Cam_", "Pred_Cam_"] + [
            "RMSE"; "MaxError"; "InPlaneError"; "OutPlaneError"
        ], ...
    1, []);
    prediction_results.results = table(...
        'Size', [size(prediction_results.data, 1), numel(results_tbl_varnames)], ...
        'VariableNames', results_tbl_varnames,...
        'VariableTypes', repmat("double", size(results_tbl_varnames))...
    );
    
    %% Process single-bend single-layer
    singlebend_singlelayer_mask = ...
        prediction_results.data.singlebend ...
        & (prediction_results.data.num_layers == 1) ...
    ;
    singlebend_singlelayer_data = prediction_results.data(...
        singlebend_singlelayer_mask, : ...
    );
    singlebend_singlelayer_pred = prediction_results.predictions(...
        singlebend_singlelayer_mask, : ...
    );
    disp("Processing single-bend single-layer results");

    for expmt_idx = 1:size(singlebend_singlelayer_pred, 1)
       % unpack data
       expmt    = singlebend_singlelayer_data.Experiment(expmt_idx);
       ins_hole = singlebend_singlelayer_data.Ins_Hole(expmt_idx);
       L_ref    = singlebend_singlelayer_data.L_ref(expmt_idx);
       L_pred   = singlebend_singlelayer_data.L_pred(expmt_idx);
       
       % unpack predictions
       kc_pred = singlebend_singlelayer_pred.kc_pred(expmt_idx);
       w_init_pred = [ 
           singlebend_singlelayer_pred.w_init_pred_1(expmt_idx);
           singlebend_singlelayer_pred.w_init_pred_2(expmt_idx);
           singlebend_singlelayer_pred.w_init_pred_3(expmt_idx);
      ];
  
       % get the curernt experiment results
       expmt_result = expmt_results_tbl(...
           expmt_results_tbl.Experiment == expmt...
           & expmt_results_tbl.Ins_Hole == ins_hole...
           & expmt_results_tbl.L_ref == L_pred ...
           , : ...
       );
       
       pos_cam      = expmt_result.cam_shape{1}';
       pos_fbg_act  = [zeros(3, 1), expmt_result.fbg_shape{1}'];
       pose_fbg_cam = expmt_result.Pose_nc{1}; 
       
       % compute predicted position
       pos_fbg_pred = singlebend_singlelayer_shape( ...
           kc_pred, w_init_pred, L_pred, needle_mechparams ...
       );
   
       % compute errors
       errors = compute_prediction_errors( pos_fbg_pred, pos_cam, pos_fbg_act, pose_fbg_cam );
       
       % tabulate errors
       indx = find( ...
           singlebend_singlelayer_data.Index(expmt_idx) == prediction_results.data.Index, ...
           1 ...
       );
       
       prediction_results.results.FBG_Pred_RMSE(indx)          = errors.pred2fbg.RMSE;
       prediction_results.results.FBG_Pred_MaxError(indx)      = errors.pred2fbg.Max;
       prediction_results.results.FBG_Pred_InPlaneError(indx)  = errors.pred2fbg.In_Plane;
       prediction_results.results.FBG_Pred_OutPlaneError(indx) = errors.pred2fbg.Out_Plane;
       
       prediction_results.results.FBG_Cam_RMSE(indx)           = errors.fbg2cam.RMSE; 
       prediction_results.results.FBG_Cam_MaxError(indx)       = errors.fbg2cam.Max;
       prediction_results.results.FBG_Cam_InPlaneError(indx)   = errors.fbg2cam.In_Plane;
       prediction_results.results.FBG_Cam_OutPlaneError(indx)  = errors.fbg2cam.Out_Plane;
       
       prediction_results.results.Pred_Cam_RMSE(indx)          = errors.pred2cam.RMSE; 
       prediction_results.results.Pred_Cam_MaxError(indx)      = errors.pred2cam.Max;
       prediction_results.results.Pred_Cam_InPlaneError(indx)  = errors.pred2cam.In_Plane;
       prediction_results.results.Pred_Cam_OutPlaneError(indx) = errors.pred2cam.Out_Plane;
    end
    
    %% Process single-bend double-layer
    singlebend_doublelayer_mask = ...
        prediction_results.data.singlebend ...
        & (prediction_results.data.num_layers == 2) ...
    ;
    singlebend_doublelayer_data = prediction_results.data(...
        singlebend_doublelayer_mask, : ...
    );
    singlebend_doublelayer_pred = prediction_results.predictions(...
        singlebend_doublelayer_mask, : ...
    );
    [sb_dl_data_uniq, data_idxs, uniq_idxs] = unique( singlebend_doublelayer_data( ...
        :, ["Experiment", "Ins_Hole", "L_ref", "L_pred"]...
    ));

    disp("Processing single-bend double-layer results");

    for expmt_idx = uniq_idxs'
       % unpack data
       expmt    = sb_dl_data_uniq.Experiment(expmt_idx);
       ins_hole = sb_dl_data_uniq.Ins_Hole(expmt_idx);
       L_ref    = sb_dl_data_uniq.L_ref(expmt_idx);
       L_pred   = sb_dl_data_uniq.L_pred(expmt_idx);
       
       % get 2-layer data
       expmt_file = fullfile(...
           needle_dir, ...
           expmt, ...
           "experiment.json" ...
       );
       z_crit = jsondecode(fileread(expmt_file)).tissue1Length;
        
       % get the curernt experiment results
       expmt_result = expmt_results_tbl(...
           expmt_results_tbl.Experiment == expmt...
           & expmt_results_tbl.Ins_Hole == ins_hole...
           & expmt_results_tbl.L_ref == L_pred ...
           , : ...
       );
       pos_cam      = expmt_result.cam_shape{1}';
       pos_fbg_act  = [zeros(3, 1), expmt_result.fbg_shape{1}'];
       pose_fbg_cam = expmt_result.Pose_nc{1}; 
       
       % unpack predictions
       pred_mask = ...
           singlebend_doublelayer_data.Experiment == expmt ...
           & singlebend_doublelayer_data.Ins_Hole == ins_hole ...
           & singlebend_doublelayer_data.L_ref == L_ref ...
           & singlebend_doublelayer_data.L_pred == L_pred ...
       ;
       kc1_pred = singlebend_doublelayer_pred{ ...
           pred_mask & singlebend_doublelayer_data.layer_num == 1, 'kc_pred' ...
       };
       kc2_pred = singlebend_doublelayer_pred{ ...
           pred_mask & singlebend_doublelayer_data.layer_num == 2, 'kc_pred' ...
       };
       w_init_pred = [ 
           singlebend_doublelayer_pred.w_init_pred_1(expmt_idx);
           singlebend_doublelayer_pred.w_init_pred_2(expmt_idx);
           singlebend_doublelayer_pred.w_init_pred_3(expmt_idx);
       ];
   
       % compute predicted shape
       pos_fbg_pred = singlebend_doublelayer_shape(...
           kc1_pred, kc2_pred, w_init_pred, L_pred, z_crit, needle_mechparams ...
       );
   
       % compute errors
       errors = compute_prediction_errors( pos_fbg_pred, pos_cam, pos_fbg_act, pose_fbg_cam );
       
       % tabulate errors
       % - index 1
       indx1_sub = singlebend_doublelayer_data{ ...
           pred_mask & singlebend_doublelayer_data.layer_num == 1, 'Index' ...
       };
       indx1 = find( ...
           indx1_sub == prediction_results.data.Index, ...
           1 ...
       );
       
       prediction_results.results.FBG_Pred_RMSE(indx1)          = errors.pred2fbg.RMSE;
       prediction_results.results.FBG_Pred_MaxError(indx1)      = errors.pred2fbg.Max;
       prediction_results.results.FBG_Pred_InPlaneError(indx1)  = errors.pred2fbg.In_Plane;
       prediction_results.results.FBG_Pred_OutPlaneError(indx1) = errors.pred2fbg.Out_Plane;
       
       prediction_results.results.FBG_Cam_RMSE(indx1)           = errors.fbg2cam.RMSE; 
       prediction_results.results.FBG_Cam_MaxError(indx1)       = errors.fbg2cam.Max;
       prediction_results.results.FBG_Cam_InPlaneError(indx1)   = errors.fbg2cam.In_Plane;
       prediction_results.results.FBG_Cam_OutPlaneError(indx1)  = errors.fbg2cam.Out_Plane;
       
       prediction_results.results.Pred_Cam_RMSE(indx1)          = errors.pred2cam.RMSE; 
       prediction_results.results.Pred_Cam_MaxError(indx1)      = errors.pred2cam.Max;
       prediction_results.results.Pred_Cam_InPlaneError(indx1)  = errors.pred2cam.In_Plane;
       prediction_results.results.Pred_Cam_OutPlaneError(indx1) = errors.pred2cam.Out_Plane;
       
       % - index 2
       indx2_sub = singlebend_doublelayer_data{ ...
           pred_mask & singlebend_doublelayer_data.layer_num == 2, 'Index' ...
       };
       indx2 = find( ...
           indx2_sub == prediction_results.data.Index, ...
           1 ...
       );
   
       prediction_results.results.FBG_Pred_RMSE(indx2)          = errors.pred2fbg.RMSE;
       prediction_results.results.FBG_Pred_MaxError(indx2)      = errors.pred2fbg.Max;
       prediction_results.results.FBG_Pred_InPlaneError(indx2)  = errors.pred2fbg.In_Plane;
       prediction_results.results.FBG_Pred_OutPlaneError(indx2) = errors.pred2fbg.Out_Plane;
       
       prediction_results.results.FBG_Cam_RMSE(indx2)           = errors.fbg2cam.RMSE; 
       prediction_results.results.FBG_Cam_MaxError(indx2)       = errors.fbg2cam.Max;
       prediction_results.results.FBG_Cam_InPlaneError(indx2)   = errors.fbg2cam.In_Plane;
       prediction_results.results.FBG_Cam_OutPlaneError(indx2)  = errors.fbg2cam.Out_Plane;
       
       prediction_results.results.Pred_Cam_RMSE(indx2)          = errors.pred2cam.RMSE; 
       prediction_results.results.Pred_Cam_MaxError(indx2)      = errors.pred2cam.Max;
       prediction_results.results.Pred_Cam_InPlaneError(indx2)  = errors.pred2cam.In_Plane;
       prediction_results.results.Pred_Cam_OutPlaneError(indx2) = errors.pred2cam.Out_Plane;
       
    end
    
    
    
    %% Process double-bend single-layer
    doublebend_singlelayer_mask = ...
        (~prediction_results.data.singlebend) ...
        & (prediction_results.data.num_layers == 1) ...
    ;
    doublebend_singlelayer_data = prediction_results.data(...
        doublebend_singlelayer_mask, : ...
    );
    doublebend_singlelayer_pred = prediction_results.predictions(...
        doublebend_singlelayer_mask, : ...
    );

    disp("Processing double-bend single-layer results");

    for expmt_idx = 1:size(doublebend_singlelayer_pred, 1)
       % unpack data
       expmt    = doublebend_singlelayer_data.Experiment(expmt_idx);
       ins_hole = doublebend_singlelayer_data.Ins_Hole(expmt_idx);
       L_ref    = doublebend_singlelayer_data.L_ref(expmt_idx);
       L_pred   = doublebend_singlelayer_data.L_pred(expmt_idx);
       
       % unpack predictions
       kc_pred = singlebend_doublelayer_pred.kc_pred(expmt_idx);
       w_init_pred = [ 
           singlebend_doublelayer_pred.w_init_pred_1(expmt_idx);
           singlebend_doublelayer_pred.w_init_pred_2(expmt_idx);
           singlebend_doublelayer_pred.w_init_pred_3(expmt_idx);
       ];
       
       % get double-bend
       expmt_file = fullfile(...
           needle_dir, ...
           expmt, ...
           "experiment.json" ...
       );
       s_dbl_bend = jsondecode(fileread(expmt_file)).DoubleBendDepth;
        
       % get the curernt experiment results
       expmt_result = expmt_results_tbl(...
           expmt_results_tbl.Experiment == expmt...
           & expmt_results_tbl.Ins_Hole == ins_hole...
           & expmt_results_tbl.L_ref == L_pred ...
           , : ...
       );
       pos_cam      = expmt_result.cam_shape{1}';
       pos_fbg_act  = [zeros(3, 1), expmt_result.fbg_shape{1}'];
       pose_fbg_cam = expmt_result.Pose_nc{1}; 
  
       % compute predicted shape
       pos_fbg_pred = doublebend_singlelayer_shape( ...
           kc_pred, w_init_pred, L_pred, s_dbl_bend, needle_mechparams ...
       );
   
       % compute errors
       errors = compute_prediction_errors( pos_fbg_pred, pos_cam, pos_fbg_act, pose_fbg_cam );
        
       % tabulate errors
       indx = find( ...
           singlebend_singlelayer_data.Index(expmt_idx) == prediction_results.data.Index, ...
           1 ...
       );
       
       prediction_results.results.FBG_Pred_RMSE(indx)          = errors.pred2fbg.RMSE;
       prediction_results.results.FBG_Pred_MaxError(indx)      = errors.pred2fbg.Max;
       prediction_results.results.FBG_Pred_InPlaneError(indx)  = errors.pred2fbg.In_Plane;
       prediction_results.results.FBG_Pred_OutPlaneError(indx) = errors.pred2fbg.Out_Plane;
       
       prediction_results.results.FBG_Cam_RMSE(indx)           = errors.fbg2cam.RMSE; 
       prediction_results.results.FBG_Cam_MaxError(indx)       = errors.fbg2cam.Max;
       prediction_results.results.FBG_Cam_InPlaneError(indx)   = errors.fbg2cam.In_Plane;
       prediction_results.results.FBG_Cam_OutPlaneError(indx)  = errors.fbg2cam.Out_Plane;
       
       prediction_results.results.Pred_Cam_RMSE(indx)          = errors.pred2cam.RMSE; 
       prediction_results.results.Pred_Cam_MaxError(indx)      = errors.pred2cam.Max;
       prediction_results.results.Pred_Cam_InPlaneError(indx)  = errors.pred2cam.In_Plane;
       prediction_results.results.Pred_Cam_OutPlaneError(indx) = errors.pred2cam.Out_Plane;
    end
    
    %% Save the results
    prediction_results_error_file = strrep( ...
        prediction_file, ".xlsx", "_errors.xlsx" ...
    );
    
    % join the tables
    prediction_result_error_tbl = horzcat(...
        prediction_results.data, ...
        prediction_results.true_predictions, ...
        prediction_results.predictions, ...
        prediction_results.results ...
    );

    % save the joint table
    writetable( ...
        prediction_result_error_tbl, prediction_results_error_file, ...
        'WriteVariableNames', true ...
    );
    fprintf("Saved results file to: %s\n", prediction_results_error_file); 

    
end


%% Helper Functions
function errors = compute_prediction_errors( ...
    pos_fbg_pred, pos_cam, pos_fbg_act, pose_fbg_cam ...
)
    errors.pred2cam = compute_camera_errors(...
        pos_cam, pos_fbg_pred, 0.5, pose_fbg_cam ...
    );
    errors.pred2fbg = compute_shape_errors(...
        pos_fbg_act, pos_fbg_pred ...
    );
    errors.fbg2cam  = compute_camera_errors(...
        pos_cam, pos_fbg_act, 0.5, pose_fbg_cam ...
    );
end

% compute errors between two shapes of the same size
function errors = compute_shape_errors(shape_ref, shape_pred)
    arguments
        shape_ref (3,:);
        shape_pred (3,:);
    end
        
    % compute distances
    devs = shape_ref - shape_pred;
    dists_sqr = dot(devs, devs, 1);
    
    errors.Min = sqrt(min(dists_sqr(2:end)));
    errors.Max = sqrt(max(dists_sqr(2:end)));
    errors.RMSE = sqrt(mean(dists_sqr(1:end)));
    errors.Tip = sqrt(dists_sqr(end));
    errors.In_Plane = mean(vecnorm(devs([2,3], :), 2,1)); % mean in-plane error
    errors.Out_Plane = mean(vecnorm(devs([1,3], :), 2,1)); % mean out-of-plane error

end

% compute errors between FBG shape and camera shape THERE IS AN ERROR HERE
function [errors, varargout] = compute_camera_errors(shape_cam, shape_fbg, ds, pose_fbg_cam)
    arguments
        shape_cam (3,:);
        shape_fbg (3,:);
        ds double = 0.5;
        pose_fbg_cam = nan;
    end
    
    % find each curve arclength
    arclen_cam = arclength(shape_cam');
    arclen_fbg = arclength(shape_fbg');
    
    % generate the arclengths
%     s_cam = 0:ds:arclen_cam;
    s_cam = flip(arclen_cam:-ds:0);
    s_fbg = 0:ds:arclen_fbg;
    
    N = min(numel(s_cam(s_cam>40)), numel(s_fbg(s_fbg > 40)));
    
    % interpolate the shapes
    shape_cam_interp = interp_pts(shape_cam', s_cam);
    shape_fbg_interp = interp_pts(shape_fbg', s_fbg);
    
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

function [pos, wv, Rmat] = singlebend_singlelayer_shape(...
    kc, w_init, L, needle_mech_params, theta0, ds ...
)
    arguments
        kc double
        w_init (3, 1)
        L double
        needle_mech_params struct
        theta0 double = 0;
        ds double = 0.5;
    end
    
    B = needle_mech_params.B;
    Binv = needle_mech_params.Binv;
    
    % needle shape
    s = 0:ds:L;
    k0 = kc * (1 - s/L).^2;
    k0_prime = -2 * kc/L * (1 - s/L);
    
    w0       = [k0;       zeros(2, length(k0))];
    w0_prime = [k0_prime; zeros(2, length(k0_prime))];
    
    [wv, pos, Rmat] = fn_intgEP_w0_Dimitri(...
        w_init, w0, w0_prime, theta0, 0, ds, length(s), B, Binv...
    );
    
    
end

function [pos, wv, Rmat] = singlebend_doublelayer_shape(...
    kc1, kc2, w_init, L, z_crit, needle_mech_params, theta0, ds ...
)
    arguments
        kc1 double
        kc2 double
        w_init (3,1)
        L double
        z_crit double
        needle_mech_params struct
        theta0 double = 0;
        ds double = 0.5;
    end
    B = needle_mech_params.B;
    Binv = needle_mech_params.Binv;
    
    s = 0:ds:L;
    
    [wv, pos, Rmat, s_crit] = fn_intgEP_zcrit_2layers_Dimitri(...
        w_init, kc1, kc2, z_crit, ...
        theta0, 0, ds, length(s), B, Binv ...
    );
    

end

function [pos, wv, Rmat] = doublebend_singlelayer_shape(...
    kc, w_init, L, s_dbl_bend, needle_mech_params, theta0, ds ...
)
    arguments
        kc double
        w_init (3,1)
        L double
        s_dbl_bend double
        needle_mech_params struct
        theta0 double = 0
        ds double = 0.5
    end
    B = needle_mech_params.B;
    Binv = needle_mech_params.Binv;
    
    if L <= s_dbl_bend
        [pos, wv, Rmat] = singlebend_singlelayer_shape(...
            kc, w_init, L, needle_mech_params, ...
            theta0, ds ...
        );
        
        if L == s_dbl_bend
            theta_z = pi;
        else
            theta_z = 0;
        end
    else
        s = 0:ds:L;
        [~, s_idx_turn] = min(abs(s - s_dbl_bend));
        s1 = s(1:s_idx_turn);
        s2 = s(s_idx_turn:end);

        kc1 = kc*((s1(end) - s1(1))/L)^(2/3);
        kc2 = kc*((s2(end) - s2(1))/L)^(2/3);

        % intrinsic curvature kappa_0 (quadratic)
        k0_1    = kc1*(1 - s1/L).^2;
        k0_2    = -kc2*(1 - s2/L).^2;
        k0_turn = (k0_1(end) + k0_2(1))/2; %0;
        k0      = [k0_1(1:end-1),k0_turn,k0_2(2:end)];

        k0_prime1     = -2*kc1/L*(1 - s1/L);
        k0_prime2     = 2*kc2/L*(1 - s2/L);
        k0_prime_peak = (k0_2(2) - k0_1(end-1))/2/ds; 
        k0_prime      = [k0_prime1(1:end-1),k0_prime_peak,k0_prime2(2:end)];
        
        % intrinsic curvature \omega_0
        w0       = [k0;       zeros(2, length(s))];
        w0_prime = [k0_prime; zeros(2, length(s))];
        
        [wv, pos, Rmat] = fn_intgEP_w0_Dimitri( ...
            w_init, w0, w0_prime, theta0, 0, ds, length(s), B, Binv ...
        );
    
        theta_z = pi;
        
    end
    
    % rotate by pi
    Rz = Rot_z(theta_z);
    pos = Rz * pos;
    Rmat = pagemtimes(Rz, Rmat);
    
end
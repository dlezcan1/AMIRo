%% 
global EXPMT_TIMEZONE
EXPMT_TIMEZONE = "America/New_York";

%% Data setup
data_dir = "/home/dlezcan1/dev/git/amiro/data/Jacynthe-Needle/2023-05-31";
resolution = "10"; % mm
l = load(...
    fullfile(...
        data_dir, ...
        sprintf('Reconstructions/reconstructions_%smm.mat', resolution) ...
     ), ...
     'Mxyz' ...
 );


ct_data_dir = fullfile(...
    data_dir, ...
    "ct_images/unpacked/results/2023-05-31" ...
);

%% Load Jacynthe's needle shapes
ref_num = 6;
jac_needle_shapes = l.Mxyz{ref_num};

skip_trials = [1, 12, 13];

%% load CT shapes
ct_data_files = dir(fullfile(ct_data_dir, "*", "ct_scan_results.xlsx"));

ct_data = cell(1, numel(ct_data_files));

for i = 1:numel(ct_data_files)
    ct_data{i} = load_ct_data(fullfile(ct_data_files(i).folder, ct_data_files(i).name));

end
ct_data = cat(1, ct_data{:});
[~, ct_idxs] = sort([ct_data.timestamp]);

%% Iterate through trials
s_ct_end_trim = 4; % mm

ds_interp = 0.5;
s_match_max = 100;
N_match_max = round(s_match_max / ds_interp);
min_reg_s = 20;
max_reg_s = nan;

AGG_NDL_PTS   = cell(1, size(jac_needle_shapes, 3));
AGG_CT_TF_PTS = cell(1, numel(AGG_NDL_PTS));

for trial_idx = 1:size(jac_needle_shapes, 3)
    if any(skip_trials == trial_idx)
        continue
    end
    ct_trial_idx = ct_idxs(trial_idx);
    fiducial_pose = ct_data(ct_trial_idx).fiducial_pose;

    ct_shape  = ct_data(ct_trial_idx).shape;
    jac_shape = jac_needle_shapes(:, :, trial_idx)';

    % transform points for CT into box frame
    ct_shape_tf = transformPointsSE3(ct_shape, finv(fiducial_pose), 2); 

    % interpolate
    [ct_shape_tf_interp, s_ct_interp] = interpolate_shape_uniform( ...
        ct_shape_tf, ...
        ds_interp, ...
        true, ...
        max_reg_s, ...
        min_reg_s ...
    );
    [jac_shape_interp, s_jac_interp] = interpolate_shape_uniform( ...
        jac_shape, ...
        ds_interp, ...
        true, ...
        max_reg_s, ...
        min_reg_s ...
    );

    % trim off 4 mm bevel-tip
    L_ct_interp = max(s_ct_interp);
    ct_shape_tf_interp = ct_shape_tf_interp(L_ct_interp - s_ct_interp >= s_ct_end_trim, : );

    % re-orient with fiducial pose
    [ct_shape_tf_interp_m, jac_shape_interp_m] = match_points( ...
        ct_shape_tf_interp,...
        jac_shape_interp, ...
        true, ...
        1, ...
        N_match_max...
    );

    AGG_NDL_PTS{trial_idx}   = jac_shape_interp_m{1};
    AGG_CT_TF_PTS{trial_idx} = ct_shape_tf_interp_m{1};

end

%% Compute registration
AGG_NDL_PTS_STK   = cat(1, AGG_NDL_PTS{:});
AGG_CT_TF_PTS_STK = cat(1, AGG_CT_TF_PTS{:});
[R_box2needle, p_box2needle] = point_cloud_reg( ...
    AGG_NDL_PTS_STK, ...
    AGG_CT_TF_PTS_STK...
);

F_box2needle = makeSE3(R_box2needle, p_box2needle, 'Check', true);

outfile = fullfile( ...
    data_dir, ...
    "reconstruction_box2needle.xlsx"...
);
writematrix(...
    F_box2needle, ...
    outfile, ...
    'Sheet', sprintf("Resolution %s mm - Ref. %d", resolution, ref_num)...
)


%% helper functions
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
function data = load_ct_data(filename)
    global EXPMT_TIMEZONE
    
    data.shape              = flip(readmatrix(filename, 'Sheet', 'needle shape'), 1);
    data.fiducial_locations = readmatrix(filename, 'Sheet', 'fiducial locations');
    data.fiducial_pose      = readmatrix(filename, 'Sheet', 'fiducial pose');
    
    % handle timestamp
    filename_split = pathsplit(filename);
    try
        data.timestamp = datetime( ...
            filename_split{end-1}, ...
            'InputFormat', 'yyyy-MM-dd_HH-mm-ss', ...
            'TimeZone', EXPMT_TIMEZONE ...
        );
    catch
        ts_str = filename_split{end};
        ts_str = ts_str(1:19);
        data.timestamp = datetime( ...
            ts_str, ...
            'InputFormat', 'yyyy-MM-dd_HH-mm-ss', ...
            'TimeZone', EXPMT_TIMEZONE ...
        );
    end
end
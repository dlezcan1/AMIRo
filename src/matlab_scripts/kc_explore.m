%% kc_explore.m
% explore kappa c versus insertion depth
data_dir = '../../data/3CH-4AA-0004/Insertion_Experiment_Results';
result_file = fullfile(data_dir, ...
    'FBG-Camera-Comp_tip-pcr_FBG-weights_combined-results.mat');
act_result_tbl = load(result_file).act_result_tbl;

act_result_tbl = renamevars(act_result_tbl, "L_ref", "L");

%% Group kc measurements into categories
% single-Layer C-Shape - 2
% single-layer S-Shape - 1
% double-layer C-Shape - 3

if ~isvariable(act_result_tbl, 'ShapeType')
    single_layer = act_result_tbl.kc2 < 0;
    c_shape = act_result_tbl.singlebend;
    
    % shape type ategories
    categ_vals = sum([single_layer, c_shape].*[1,2], 2);
    shape_type = categorical(categ_vals, [1,2,3],...
        {'1-Layer S-Shape', '2-Layer C-Shape', '1-Layer C-Shape'});
    
    act_result_tbl.ShapeType(:) = shape_type;
    
    unique(act_result_tbl.ShapeType)
end

%% Determine kc_ref and L_ref
L_min = 30; % minimum reference insertion depth (for cut-off)
mask_L_min = act_result_tbl.L >= L_min;
mask_c_1layer = act_result_tbl.ShapeType == "1-Layer C-Shape";
mask_s_1layer = act_result_tbl.ShapeType == "1-Layer S-Shape";
mask_c_2layer = act_result_tbl.ShapeType == "2-Layer C-Shape";

% reference values (uniitalized)
act_result_tbl.L_ref(:) = -1; 
act_result_tbl.kc1_ref(:) = -1;
act_result_tbl.kc2_ref(:) = -1;

% iterate through experiments
for expmt = unique(act_result_tbl.Experiment)'
    disp(expmt + ":")
    mask_expmt = act_result_tbl.Experiment == expmt;
    for ins_hole = unique(act_result_tbl.Ins_Hole)'
        fprintf("Insertion Hole: %d... ", ins_hole);
        mask_ins_hole = act_result_tbl.Ins_Hole == ins_hole;
        
        % determine ref. for 1-layer C-shape
        mask1 = mask_expmt & mask_ins_hole & mask_L_min & ...
            mask_c_1layer;
        L_ref_c_1layer = min(act_result_tbl.L(mask1));
        if ~isempty(L_ref_c_1layer)
            mask_L_ref = act_result_tbl.L == L_ref_c_1layer;
            kc1_ref = act_result_tbl.kc1(mask1 & mask_L_ref);
            kc2_ref = act_result_tbl.kc2(mask1 & mask_L_ref);

            % add to table
            mask_add = mask_expmt & mask_ins_hole & mask_c_1layer;
            act_result_tbl.L_ref(mask_add) = L_ref_c_1layer;
            act_result_tbl.kc1_ref(mask_add) = kc1_ref;
            act_result_tbl.kc2_ref(mask_add) = kc2_ref;
        end

        % determine ref. for 1-layer S-shape
        mask2 = mask_expmt & mask_ins_hole & mask_L_min & ...
            mask_s_1layer;
        L_ref_s_1layer = min(act_result_tbl.L(mask2));
        if ~isempty(L_ref_s_1layer)
            mask_L_ref = act_result_tbl.L == L_ref_s_1layer;
            kc1_ref = act_result_tbl.kc1(mask2 & mask_L_ref);
            kc2_ref = act_result_tbl.kc2(mask2 & mask_L_ref);

            % add to table
            mask_add = mask_expmt & mask_ins_hole & mask_s_1layer;
            act_result_tbl.L_ref(mask_add) = L_ref_s_1layer;
            act_result_tbl.kc1_ref(mask_add) = kc1_ref;
            act_result_tbl.kc2_ref(mask_add) = kc2_ref;
        end

        % determine ref. for 2-layer C-shape
        mask3 = mask_expmt & mask_ins_hole & mask_L_min & ...
            mask_c_2layer;
        L_ref_c_2layer = min(act_result_tbl.L(mask3));
        if ~isempty(L_ref_c_2layer)
            mask_L_ref = act_result_tbl.L == L_ref_c_2layer;
            kc1_ref = act_result_tbl.kc1(mask3 & mask_L_ref);
            kc2_ref = act_result_tbl.kc1(mask3 & mask_L_ref);

            % add to table
            mask_add = mask_expmt & mask_ins_hole & mask_c_2layer;
            act_result_tbl.L_ref(mask_add) = L_ref_c_2layer;
            act_result_tbl.kc1_ref(mask_add) = kc1_ref;
            act_result_tbl.kc2_ref(mask_add) = kc2_ref;
        end

        disp("Completed")
    end
    
    disp(" ")
end

%% Calculate kc/kc_ref and L/L_ref
act_result_tbl.kc1_ratio = act_result_tbl.kc1 ./ act_result_tbl.kc1_ref .* ...
    sign(act_result_tbl.kc1);
act_result_tbl.kc2_ratio = act_result_tbl.kc2 ./ act_result_tbl.kc2_ref .* ...
    sign(act_result_tbl.kc2);
act_result_tbl.L_ratio = act_result_tbl.L./act_result_tbl.L_ref;

%% Plotting
% plotting masks
mask_c_1layer = act_result_tbl.ShapeType == "1-Layer C-Shape";
mask_s_1layer = act_result_tbl.ShapeType == "1-Layer S-Shape";
mask_c_2layer = act_result_tbl.ShapeType == "2-Layer C-Shape";

mask_soft1 = act_result_tbl.TissueStiffness1 <= 29;
mask_hard1 = ~mask_soft1;

mask_hard2 = (act_result_tbl.TissueStiffness2 > 29);
mask_soft2 = ~mask_hard2 & (act_result_tbl.TissueStiffness2 > 0);

% - minimum insertion depth mask
mask_Lmin = act_result_tbl.L >= 30;
mask_c_1layer = mask_c_1layer & mask_Lmin;
mask_s_1layer = mask_s_1layer & mask_Lmin;
mask_c_2layer = mask_c_2layer & mask_Lmin;

mask_c_2layer_kc2 = mask_c_2layer & (act_result_tbl.kc2_ratio >= 0);

fig = figure(1);
set(fig, 'Units', 'normalized', 'Position', [0, 0.1, 1,0.8]);
hold off;
plot(act_result_tbl.L_ratio(mask_c_1layer & mask_soft1), log(act_result_tbl.kc1_ratio(mask_c_1layer & mask_soft1)), ...
    '.', 'Markersize', 12, 'DisplayName', '1-Layer C-shape: Soft \kappa_c'); hold on;

plot(act_result_tbl.L_ratio(mask_c_1layer & mask_hard1), log(act_result_tbl.kc1_ratio(mask_c_1layer & mask_hard1)), ...
    '.', 'Markersize', 12, 'DisplayName', '1-Layer C-shape: Hard \kappa_c'); hold on;


plot(act_result_tbl.L_ratio(mask_c_2layer & mask_soft1), log(act_result_tbl.kc1_ratio(mask_c_2layer & mask_soft1)), ...
    '.', 'Markersize', 12, 'DisplayName', '2-Layer C-shape: Soft \kappa_{c,1}'); hold on;

plot(act_result_tbl.L_ratio(mask_c_2layer & mask_hard1), log(act_result_tbl.kc1_ratio(mask_c_2layer & mask_hard1)), ...
    '.', 'Markersize', 12, 'DisplayName', '2-Layer C-shape: Hard \kappa_{c,1}'); hold on;


plot(act_result_tbl.L_ratio(mask_c_2layer_kc2 & mask_soft2), log(act_result_tbl.kc2_ratio(mask_c_2layer_kc2 & mask_soft2)), ...
    '.', 'Markersize', 12, 'DisplayName', '2-Layer C-shape: Soft \kappa_{c,2}'); hold on;

plot(act_result_tbl.L_ratio(mask_c_2layer_kc2 & mask_hard2), log(act_result_tbl.kc2_ratio(mask_c_2layer_kc2 & mask_hard2)), ...
    '.', 'Markersize', 12, 'DisplayName', '2-Layer C-shape: Hard \kappa_{c,2}'); hold on;


plot(act_result_tbl.L_ratio(mask_s_1layer & mask_soft1), log(act_result_tbl.kc1_ratio(mask_s_1layer & mask_soft1)), ...
    '.', 'Markersize', 12, 'DisplayName', '1-Layer S-shape: Soft \kappa_c'); hold on;

plot(act_result_tbl.L_ratio(mask_s_1layer & mask_hard1), log(act_result_tbl.kc1_ratio(mask_s_1layer & mask_hard1)), ...
    '.', 'Markersize', 12, 'DisplayName', '1-Layer S-shape: Hard \kappa_c'); hold on;

xlabel('L / L_{ref}'); ylabel('log( \kappa_c / \kappa_{c,ref} )'); 
xlim([0, max(xlim)]);
legend('location', 'northeastoutside')
title("\kappa_c Log-Ratio vs Insertion Depth Ratio");
grid on;

%% p-value fitting
fit_soft1 = polyfit(act_result_tbl.L_ratio(mask_c_1layer & mask_soft1), ...
                    log(act_result_tbl.kc1_ratio(mask_c_1layer & mask_soft1)),...
                    1)
p_polyfit_soft1 = fit_soft1(1)
p_lsqr_soft1 = act_result_tbl.L_ratio(mask_c_1layer & mask_soft1) \ ...
               log(act_result_tbl.kc1_ratio(mask_c_1layer & mask_soft1))

fit_hard1 = polyfit(act_result_tbl.L_ratio(mask_c_1layer & mask_hard1), ...
                    log(act_result_tbl.kc1_ratio(mask_c_1layer & mask_hard1)),...
                    1)
p_hard1 = fit_hard1(1)
p_lsqr_hard1 = act_result_tbl.L_ratio(mask_c_1layer & mask_hard1) \ ...
               log(act_result_tbl.kc1_ratio(mask_c_1layer & mask_hard1))

%% Saving

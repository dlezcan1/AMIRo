% update plots lateral deflection
%% Plot Defaults
% variables
linewidth     = 6;
markersize    = 20;
fontsize      = 26;
titlefontsize = round(1.5*fontsize);
fontweight    = 'bold';
bp_linewidth   = 3;
axlbl_fontsize = 20;

% defaults
set(0, 'DefaultLineLineWidth', linewidth);
set(0, 'DefaultAxesFontSize', fontsize);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontWeight', 'bold');
set(0, 'DefaultTextFontSize', round(fontsize * 1.5));

%% Data file
pub_dir = fullfile( ...
    "/mnt/data-drive/onedrive/Publications/2024 EMBC" ...
);
data_dir = fullfile( ...
    pub_dir, "Data" ...
);
figure_dir = fullfile( ...
    pub_dir, "Figures" ...
);

datafile_fmt = "all_experiment_results_%s_analyzed.mat";
alldata_fmt = "all";

airgap_length = 45; % mm

%% Load the data
deflection_column_name = "deflection-direction";
if ~isfile(fullfile(data_dir, sprintf(datafile_fmt, alldata_fmt)))
    % in-bevel direction
    in_bevel_tbl = load_variable( ...
        fullfile( ...
            data_dir, ...
            sprintf(datafile_fmt, 'in-bevel')...
        ), ...
        'expmt_results' ...
    );
    in_bevel_tbl{:, deflection_column_name} = "in-bevel";
    
    % out-bevel direction
    out_bevel_tbl = load_variable(...
        fullfile( ...
            data_dir, ...
            sprintf(datafile_fmt, 'out-bevel')...
        ), ...
        'expmt_results' ...
    );
    out_bevel_tbl{:, deflection_column_name} = "out-bevel";
    
    all_tbl = [in_bevel_tbl; out_bevel_tbl];

    all_tbl.tissue_length(:) = airgap_length;
    all_tbl.insertion_depth = all_tbl.insertion_depth - all_tbl.tissue_length;
    
    save(...
        fullfile( ...
            data_dir, ...
            sprintf(datafile_fmt, 'all')...
        ), ...
        'all_tbl'...
    );
else
    all_tbl = load_variable( ...
        fullfile( ...
            data_dir, ...
            sprintf(datafile_fmt, alldata_fmt)...
        ), ...
        'all_tbl'...
    );
end

%% Group summarize the results
error_columns = all_tbl.Properties.VariableNames( ...
    endsWith(all_tbl.Properties.VariableNames, '_Error') ...
    | strcmp(all_tbl.Properties.VariableNames, 'RMSE') ...
);
all_tbl_summ = groupsummary(...
    all_tbl, ...
    {char(deflection_column_name), 'insertion_depth'},...
    {'mean', 'std'}, ...
    error_columns...
)

outfile = fullfile(data_dir, strrep(sprintf(datafile_fmt, 'error-stats'), '.mat', '.csv'));
writetable(...
    all_tbl_summ, ...
    outfile ...
);
fprintf("Saved summary statistics to: %s\n", outfile)

%% Plot the statistics
plot_errors = error_columns(...
    ~contains(error_columns, 'Min') ...
    & ~contains(error_columns, 'Tip') ...
);


fig_bp = figure(1);
set(fig_bp, 'Units', 'normalized', 'position', [0.05, -0.05, 0.8, 0.9]);
fig_dep = figure(2);
set(fig_dep, 'Units', 'normalized', 'position', [0.1, -0.05, 0.8, 0.9]);
fig_both = figure(3);
set(fig_both, 'Units', 'normalized', 'position', [0.15, -0.05, 0.8, 0.9]);

% plots
max_yl = -inf;
for i = 1:numel(plot_errors)
    error_i = plot_errors{i};
    
    [c, r] = ind2sub([2, 2], i);

    % deflection direction
    figure(fig_bp);
    ax_bp = subplot(2, 2, i);
    
    
    boxplot( ...
        all_tbl.(error_i), ...
        all_tbl.(deflection_column_name) ...
    )
    title(strrep(error_i, '_', ' '))
    ax_bp.TickLabelInterpreter = 'tex';

    % -update ylimit
    max_yl = max(max_yl, max(ax_bp.YLim));    
    
    if c == 1
        ylabel("Error (mm)")
    end

    % insertion depth
    figure(fig_dep)
    ax_dep = subplot(2, 2, i);
    
    mask_depth = all_tbl.insertion_depth > 0;    
    boxplot( ...
        all_tbl{mask_depth, error_i}, ...
        all_tbl.insertion_depth(mask_depth) ...
    );
    title(strrep(error_i, '_', ' '))
    ax_dep.TickLabelInterpreter = 'tex';

    % -  update ylimit
    max_yl = max(max_yl, max(ax_dep.YLim));    
    
    if c == 1
        ylabel("Error (mm)")
    end

    % insertion depth and deflection direction
    figure(fig_both)
    ax_both = subplot(2, 2, i);

    [~, sidxs] = sort(strcat(all_tbl.(deflection_column_name), num2str(all_tbl.insertion_depth)));
    all_tbl_sorted = all_tbl(sidxs, :);
    mask_depth_std = all_tbl_sorted.insertion_depth > 0;

    boxplot( ...
        all_tbl_sorted{mask_depth_std, error_i}, ...
        strcat( ...
            num2str(all_tbl_sorted.insertion_depth(mask_depth_std)), ...
            "\newline" , ...
            strrep(all_tbl_sorted{mask_depth_std, deflection_column_name}, '-bevel', '')...
        ) ...
    );
    title(strrep(error_i, '_', ' '))
    ax_both.TickLabelInterpreter = 'tex';
    

    % - update ylimit
    max_yl = max(max_yl, max(ax_both.YLim));    
    
    if c == 1
        ylabel("Error (mm)")
    end

end


%% y-limits
for i = 1:numel(plot_errors)
    fig_bp.Children(i).YLim = [0, max_yl];

    fig_dep.Children(i).YLim = [0, max_yl];

    fig_both.Children(i).YLim = [0, max_yl];
    
    fig_both.Children(i).Position(4) = 0.35;
    if i == 1 || i == 2
        fig_both.Children(i).Position(2) = 0.10;
    else
        fig_both.Children(i).Position(2) = 0.60;
    end

end

for fig = [fig_bp, fig_dep, fig_both]
    set(findobj(fig, 'type', 'line'), 'LineWidth', bp_linewidth)
    set(findall(fig, 'type', 'axes'), 'FontSize', axlbl_fontsize)
    set(findall(fig_bp, '-property', 'FontSize', '-not', 'type', 'axes'), 'FontSize', axlbl_fontsize);
end


%% save the figure
savefigas( ...
    fig_bp, ...
    fullfile( ...
        figure_dir, ...
        "error-statistics-deflection_direction"...
    ), ...
    'Verbose', 'true' ...
)
savefigas( ...
    fig_dep, ...
    fullfile( ...
        figure_dir, ...
        "error-statistics-insdepth"...
    ), ...
    'Verbose', 'true' ...
)
savefigas( ...
    fig_both, ...
    fullfile( ...
        figure_dir, ...
        "error-statistics-deflection_direction_insdepth"...
    ), ...
    'Verbose', 'true' ...
)
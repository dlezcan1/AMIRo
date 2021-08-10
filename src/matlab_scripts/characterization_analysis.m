%% characterization_analysis.m
%
% function to analyze characterization data
%
% - written by: Dimitri Lezcano

function characterization_analysis(data_dir, num_ch, num_aa, num_trials, depths, ch_remap)
    arguments
        data_dir    string;        
        num_ch      double {mustBeInteger, mustBePositive};
        num_aa      double {mustBeInteger, mustBePositive};
        num_trials  double {mustBeInteger, mustBePositive};
        depths      (1,:)  {mustBeNonnegative};
        ch_remap    (1,:)  {mustBeInteger, mustBePositive} = 1:num_ch*num_aa;
    end
    %% Set-up
    set(0, 'defaultLineLineWidth', 1);
    
    % initalizations
    data_files = struct();
    ch_names = cell(1,num_ch);
    
    % Formatted names
    sheet_name  = 'trial%d%dmm';
    load_name   = "CH%dload.xls";
    unload_name = "CH%dunload.xls";
    
    % file names
    data_tbl_out_name    = "Entire_Data";
    trial_fig_name       = "tr%dCH%dAA%d";
    mean_ch_aa_fig_name  = "mean_of_CH%d_AA%d";
    mean_ch_fig_name     = "mean_of_CH%d";
    
    % gather the data files and values
    for ch_i = 1:num_ch
        CHi = strcat("CH", num2str(ch_i));
        data_files.(CHi).load = fullfile(data_dir,sprintf(load_name, ch_i));
        data_files.(CHi).unload = fullfile(data_dir,sprintf(unload_name, ch_i));
        ch_names{ch_i} = CHi;
    end
    
    % set-up data table (trial_num, depth, ch, load?, wavelengths
    ch_aa_pairs = cart_product((1:num_ch)', (1:num_aa)');
    ch_aa_names = split(strip(sprintf("CH%d_AA%d ", ch_aa_pairs'), 'right', ' '), ' ', 2);
    var_types = ["uint8", "double", "uint8", "categorical", repmat("double", 1, num_ch*num_aa)];
    var_names = ["trial", "depth", "CH", "load", ch_aa_names];
    data_tbl = table('Size', [0,4 + num_ch*num_aa], 'VariableTypes', var_types,...
                    'VariableNames', var_names);
                    
    
    %% Gather data
    if exist(append(fullfile(data_dir, data_tbl_out_name), '.mat'), 'file') == 2
        data_tbl = load(append(fullfile(data_dir, data_tbl_out_name), '.mat'), 'data_tbl').data_tbl;
        disp("Data table already parsed and loaded from file.");
        
    else
        warning('off');
        for ch_i = 1:num_ch
            fprintf('Channel: %d\n', ch_i);
            CHi = ch_names{ch_i};
            for trial = 1:num_trials
               fprintf("Trial: %d\n", trial);
               for d = depths
                   fprintf('Depth: %.1f mm\n', d);

                   sheet_name_d = sprintf(sheet_name, trial, d);
                   
                   if d == max(depths) % only do load 
                       % read in data table
                       load_mat   = readmatrix(data_files.(CHi).load,   'Sheet', sheet_name_d);
                       
                       % remap and resave the wavelengths
                       if any(ch_remap ~= 1:num_ch*num_aa)
                           load_mat   = load_mat(:,ch_remap);
                           
                           writematrix(load_mat, data_files.(CHi).load, 'Sheet', sheet_name_d);
                           disp("Rewrote load matrix for channel remapping")
                       end
                       
                       % get the mean wavelengths
                       load_wls   = mean(load_mat);                       

                       % append the values to the table
                       data_tbl{end+1, 1:3} = [trial, d, ch_i]; %#ok<AGROW>
                                           
                       % load values
                       data_tbl.load(end)                    = 'load';
                       data_tbl{end,end-num_ch*num_aa+1:end} = load_wls;

                   else
                       % read in data table
                       load_mat   = readmatrix(data_files.(CHi).load,   'Sheet', sheet_name_d);
                       unload_mat = readmatrix(data_files.(CHi).unload, 'Sheet', sheet_name_d);
                       
                       % remap and resave the wavelengths
                       if any(ch_remap ~= 1:num_ch*num_aa)
                           load_mat   = load_mat(:,ch_remap);
                           unload_mat = unload_mat(:,ch_remap);
                           
                           writematrix(load_mat, data_files.(CHi).load, 'Sheet', sheet_name_d);
                           disp("Rewrote load matrix for channel remapping")
                           
                           writematrix(unload_mat, data_files.(CHi).unload, 'Sheet', sheet_name_d);
                           disp("Rewrote unload matrix for channel remapping")
                       end
                       
                       % get the mean wavelengths
                       load_wls   = mean(load_mat);
                       unload_wls = mean(unload_mat);

                       % append the values to the table
                       data_tbl{end+1:end+2, 1:3} = [trial, d, ch_i;
                                                     trial, d, ch_i];

                       % load values
                       data_tbl.load(end-1)                    = 'load';
                       data_tbl{end-1,end-num_ch*num_aa+1:end} = load_wls;

                       % unload values
                       data_tbl.load(end)                      = 'unload';
                       data_tbl{end,end-num_ch*num_aa+1:end}   = unload_wls;
                   end
               end
               fprintf("End of trial\n\n");
            end
            
            disp(' ');
        end
        warning('on');

        % save the data
        fileout = fullfile(data_dir, data_tbl_out_name);
        save(append(fileout, '.mat'), 'data_tbl');
        fprintf("Saved data file %s\n", append(fileout, '.mat'));
        writetable(data_tbl, append(fileout, '.xlsx'), 'WriteMode', 'overwritesheet');
        fprintf("Saved data file %s\n", append(fileout, '.xlsx'));
    end
    disp(' ');

    %% Begin analysis
    % analysis of individual trials 
    for ch_i = 1:num_ch
        for trial = 1:num_trials
            sub_tbl = data_tbl(data_tbl.trial == trial & data_tbl.CH == ch_i,:);

            load_sub_tbl   = sub_tbl(sub_tbl.load=='load',:);
            unload_sub_tbl = sub_tbl(sub_tbl.load=='unload',:);
            load_d         = load_sub_tbl.depth;
            unload_d       = unload_sub_tbl.depth;
            [~, unload_idxs] = sort(unload_d);
            ref_wl_tbl     = load_sub_tbl(load_sub_tbl.depth==min(load_d),:);
            
            trial_fig = figure(1);
            for aa_i = 1:num_aa
               ch_aa     = sprintf('CH%d_AA%d', ch_i, aa_i);
               wl_load   = load_sub_tbl.(ch_aa) - ref_wl_tbl.(ch_aa);
               wl_unload = unload_sub_tbl.(ch_aa) - ref_wl_tbl.(ch_aa);
               
               plot(load_d,   wl_load,   '*-', 'DisplayName', 'load'); hold on;
               plot(unload_d(unload_idxs), wl_unload(unload_idxs), '*-', 'DisplayName', 'unload'); hold on;
               hold off;
               legend('Location', 'eastoutside');
               title(sprintf("CH%d AA%d load & unload", ch_i, aa_i));
               xlabel('tip deflection (mm)');
               ylabel('wavelength shift (nm)');
               
               fileout = append(sprintf(trial_fig_name, trial, ch_i, aa_i),'.png');
               fileout = fullfile(data_dir, fileout);
               saveas(trial_fig, fileout);
               fprintf("Saved figure: %s\n", fileout);
            end
        end
    end
    
    % analysis of mean active areas per channel
    grouped_trials = grpstats(data_tbl, {'CH', 'load', 'depth'}, {'mean', 'std'},...
        'DataVars', data_tbl.Properties.VariableNames(end-num_ch*num_aa+1:end));
    
    ch_fig = figure(2);
    for ch_i = 1:num_ch
        sub_stats  = grouped_trials(grouped_trials.CH == ch_i,:);
        sub_load_stats = sub_stats(sub_stats.load == 'load',:);
        sub_unload_stats = sub_stats(sub_stats.load == 'unload',:);
        ref_wl_tbl = data_tbl(data_tbl.CH == ch_i & data_tbl.depth == min(data_tbl.depth) & ...
                              data_tbl.load == 'load' & data_tbl.trial==1, :);
        for aa_i = 1:num_aa
            ch_aa = sprintf('CH%d_AA%d', ch_i, aa_i);
            
            % gather data values from the table for plotting
            d_load       = sub_load_stats.depth;
            d_unload     = sub_unload_stats.depth;
            wl_load_aa   = sub_load_stats.(strcat('mean_', ch_aa)) - ref_wl_tbl.(ch_aa);
            wl_unload_aa = sub_unload_stats.(strcat('mean_', ch_aa)) - ref_wl_tbl.(ch_aa);
            std_load_aa  = sub_load_stats.(strcat('std_', ch_aa));
            std_unload_aa  = sub_unload_stats.(strcat('std_', ch_aa));
            
            d      = [d_load;      d_unload(unload_idxs)];
            wl_aa  = [wl_load_aa;  wl_unload_aa(unload_idxs)];
            std_aa = [std_load_aa; std_unload_aa(unload_idxs)];
            
            % plot the means and wavelengths
            errorbar(d, wl_aa, std_aa, '*-', 'DisplayName', sprintf("AA%d", aa_i)); hold on;
            
        end
        
        hold off;
        xlabel('tip deflection (mm)'); 
        ylabel('wavelength shift (nm)');
        title(sprintf("CH%d load & unload: mean of %d trials", ch_i, num_trials));
        legend('Location', 'southoutside', 'Orientation', 'horizontal');
        
        fileout = append(sprintf(mean_ch_fig_name, ch_i), '.png');
        fileout = fullfile(data_dir, fileout);
        saveas(ch_fig, fileout);
        fprintf("Saved figure: %s\n", fileout);
        
    end
    
    % analysis of mean channels per active area
    for ch_i = 1:num_ch
        sub_stats  = grouped_trials(grouped_trials.CH == ch_i,:);
        sub_load_stats = sub_stats(sub_stats.load == 'load',:);
        sub_unload_stats = sub_stats(sub_stats.load == 'unload',:);
        ref_wl_tbl = data_tbl(data_tbl.CH == ch_i & data_tbl.depth == min(data_tbl.depth) & ...
                              data_tbl.load == 'load' & data_tbl.trial==1, :);
        aa_fig = figure(3);
        for aa_i = 1:num_aa
            for ch_j = 1:num_ch
                ch_aa = sprintf("CH%d_AA%d",ch_j, aa_i);
                
                % gather data values from the table for plotting
                d_load       = sub_load_stats.depth;
                d_unload     = sub_unload_stats.depth;
                wl_load_aa   = sub_load_stats.(strcat('mean_', ch_aa)) - ref_wl_tbl.(ch_aa);
                wl_unload_aa = sub_unload_stats.(strcat('mean_', ch_aa)) - ref_wl_tbl.(ch_aa);
                std_load_aa  = sub_load_stats.(strcat('std_', ch_aa));
                std_unload_aa  = sub_unload_stats.(strcat('std_', ch_aa));
                
                [~, unload_idxs] = sort(d_unload,'descend');
                d      = [d_load;      d_unload(unload_idxs)];
                wl_aa  = [wl_load_aa;  wl_unload_aa(unload_idxs)];
                std_aa = [std_load_aa; std_unload_aa(unload_idxs)];
                
                
                % plot data
                errorbar(d, wl_aa, std_aa, '*-', 'DisplayName', sprintf("CH%d", ch_j)); hold on;
                
            end
            hold off;
            xlabel('tip deflection (mm)'); 
            ylabel('wavelength shift (nm)');
            title(sprintf("CH%d load & unload: mean of %d trials for AA%d", ch_i, num_trials, aa_i));
            legend('Location', 'southoutside', 'Orientation', 'horizontal');
            
            % save the figure
            fileout = append(sprintf(mean_ch_aa_fig_name, ch_i, aa_i),'.png');
            fileout = fullfile(data_dir, fileout);
            saveas(aa_fig, fileout);
            fprintf("Saved figure: %s\n", fileout);
        end
    end
end
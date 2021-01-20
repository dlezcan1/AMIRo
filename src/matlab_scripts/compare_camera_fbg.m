%% compare_camera_fbg.m
% 
% this is a script to compare shape sensing methods to FBG shape sensing
%
% - written by: Dimitri Lezcano

%% Set-Up
% directories to iterate throughn ( the inidividual trials )
expmt_dir = "../../data/needle_3CH_3AA/01-18-2021_Test-Insertion-Expmt/";
trial_dirs = dir(expmt_dir + "Insertion*/");
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% saving options
save_bool = true;
fileout_base = "Jig-Camera-Comp";

% directory separation
if ispc
    dir_sep = '\';
else
    dir_sep = '/';
end

% 3D point file names
camera_pos_file = "left-right_3d-pts.txt";
fbg_pos_file = "FBGdata_3d-position.xls";

% arclength options
ds = 0.5;


%% Process each trial
for i = 1:length(trial_dirs)
    tic; 
    % trial operations
    L = str2double(trial_dirs(i).name);
    re_ret = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
    hole_num = str2double(re_ret{1}{1});
    
    % trial directory
    d = strcat(trial_dirs(i).folder,dir_sep, trial_dirs(i).name, dir_sep);
    fbg_file = d + fbg_pos_file;
    camera_file = d + camera_pos_file;
    
    % load in the matrices
    fbg_pos = readmatrix(fbg_file)';
    camera_pos = readmatrix(camera_file);
    camera_pos = camera_pos(:,1:3);
    
    % get the arclengths of each curve
    arclen_fbg = arclength(fbg_pos);
    arclen_camera = arclength(camera_pos);
    
    fprintf("Arclengths (actual, FBG, Camera) [mm]: %.2f, %.2f, %.2f\n", L, arclen_fbg, arclen_camera);
    
    % interpolate both points for correspondence
    s_fbg = 0:ds:arclen_fbg;
    s_camera = 0:ds:arclen_camera;
    if length(s_fbg) == length(s_camera)
        s_max = s_fbg;
    elseif length(s_fbg) > length(s_camera)
        s_max = s_fbg;
    else
        s_max = s_camera;
    end
    N = min(length(s_fbg), length(s_camera)); % minimum number of points to match
    
    fbg_pos_interp = interp_pts(fbg_pos, s_fbg);
    camera_pos_interp = interp_pts(camera_pos, s_camera);
    
    % align the points: camera aligned -> fbg coordinate system
    [R, p] = point_cloud_reg_tip(camera_pos_interp(end-N+1:end,:),... 
                                 fbg_pos_interp(end-N+1:end,:));
    
    camera_pos_interp_tf = camera_pos_interp * R' + p';
    
    % error analysis
    errors = error_analysis(camera_pos_interp_tf(end-N+1:end,:),...
                            fbg_pos_interp(end-N+1:end,:));
    
    % Plotting
    %- 3-D shape 
    fig_shape_3d = figure(1);
    set(fig_shape_3d,'units','normalized','position', [0, 0.4, 1/3, .5])
    plot3(fbg_pos_interp(:,3), fbg_pos_interp(:,1), fbg_pos_interp(:,2), 'g-', 'LineWidth', 2); hold on;
    plot3(camera_pos_interp_tf(:,3), camera_pos_interp_tf(:,1), camera_pos_interp_tf(:,2), 'r-',...
        'LineWidth', 2); 
    hold off;
    legend('FBG', 'Stereo Recons.'); 
    title(sprintf('3-D shapes: Hole Num=%d, Ins. Depth=%.1f mm', hole_num, L));
    xlabel('z','fontweight', 'bold'); ylabel('x','fontweight', 'bold'); 
    zlabel('y','fontweight', 'bold');
    axis equal; grid on;
    
    
    %- 2-D shape
    fig_shape_2d = figure(2);
    set(fig_shape_2d,'units','normalized','position', [1/3, 0.4, 1/3, .5] )
    %-- in-plane
    subplot(2,1,1);
    plot(fbg_pos_interp(:,3), fbg_pos_interp(:,2), 'g-', 'LineWidth', 2); hold on;
    plot(camera_pos_interp_tf(:,3), camera_pos_interp_tf(:,2), 'r-', 'LineWidth', 2);
    hold off;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    %-- out-of-plane
    subplot(2,1,2);
    plot(fbg_pos_interp(:,3), fbg_pos_interp(:,1), 'g-', 'LineWidth', 2); hold on;
    plot(camera_pos_interp_tf(:,3), camera_pos_interp_tf(:,1), 'r-', 'LineWidth', 2);
    hold off;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    legend('FBG', 'Stereo Recons.'); 
    
    sgtitle(sprintf('2-D shapes: Hole Num=%d, Ins. Depth=%.1f mm', hole_num, L));
    
    %- error plots
    fig_err = figure(3);
    s_sub = s_max(end-N+1:end);
    set(fig_err,'units','normalized','position', [2/3, 0.4, 1/3, .5])
    plot(s_max, 0.5 * ones(size(s_max)), 'r--', 'DisplayName', '0.5 mm'); hold on;
    plot(s_sub, errors.L2, 'DisplayName', 'L2 Distance'); 
    plot(s_sub, errors.dx, 'DisplayName', 'x-component');
    plot(s_sub, errors.dy, 'DisplayName', 'y-component');
    plot(s_sub, errors.dz, 'DisplayName', 'z-component');
    plot(s_sub, errors.in_plane, 'DisplayName', 'in-plane');
    plot(s_sub, errors.out_plane, 'DisplayName', 'out-of-plane');
    hold off;
    xlabel('s [mm]', 'fontweight', 'bold');
    ylabel('error [mm]','fontweight', 'bold');
    xlim([0, 1.1*max(s_max)]); ylim([0, max([1.1 * errors.L2', 1])]);
    legend(); grid on;
    title(sprintf('Errors: Hole Num=%d, Ins. Depth=%.1f mm', hole_num, L));
    
    % time update
    t = toc;
    
    % saving
    if save_bool
        % write arclengths from each position
        T = table(L, arclen_fbg, arclen_camera, 'VariableNames', {'L', 'FBG', 'Camera'});
        writetable(T, d + fileout_base + "_arclengths-mm.txt");
        fprintf("Wrote arclengths to: '%s'\n", d + fileout_base + "_arclengths-mm.txt");
       
        % write the figures
        %- 3-D plot
        verbose_savefig(fig_shape_3d, d + fileout_base + "3d-positions.fig");
        verbose_saveas(fig_shape_3d, d + fileout_base + "3d-positions.png");
        
        %- 2-D plot
        verbose_savefig(fig_shape_2d, d + fileout_base + "2d-positions.fig");
        verbose_saveas(fig_shape_2d, d + fileout_base + "2d-positions.png");
        
        %- error plot
        verbose_savefig(fig_err, d + fileout_base + "3d-positions-errors.fig");
        verbose_saveas(fig_err, d + fileout_base + "3d-positions-errors.png");
        
    end
    
    % update user
    fprintf("Finished trial: '%s' in %.2f secs.\n", d, t);
    disp(" ");
    
end

%% Helper functions
% simple arclength integration
function [L, varargout] = arclength(pts)
    dpts = diff(pts, 1, 1); % pts[i+1] - pts[i]
    
    dl = vecnorm(dpts, 2, 2); % ||dpts||
    
    L = sum(dl);
    
    varargout{1} = dl; % ds
    varargout{2} = [0; cumsum(dl)]; % s
    
end

% interpolate nurbs-pts for standardized ds
function [pts_std, varargout] = interp_pts(pts, s_interp)
    [~, ~, s_lu] = arclength(pts);
    
    % look-up for interpolation
    x_lu = pts(:,1); 
    y_lu = pts(:,2); 
    z_lu = pts(:,3);
    
    % interpolation
    x_interp = interp1(s_lu, x_lu, s_interp);
    y_interp = interp1(s_lu, y_lu, s_interp);
    z_interp = interp1(s_lu, z_lu, s_interp);
    
    % combine the output
    pts_std = [x_interp', y_interp', z_interp'];
    varargout{1} = s_interp; % return the interpoation arclength just in case

end

% error analysis
function errors = error_analysis(nurbs, jig)
% measures error metrics from each points
    
    % L2 distance
    errors.L2 = vecnorm(nurbs - jig, 2, 2); 
    
    % component-wise error
    errors.dx = abs(nurbs(:,1) - jig(:,1));
    errors.dy = abs(nurbs(:,2) - jig(:,2));
    errors.dz = abs(nurbs(:,3) - jig(:,3));
    
    % in/out-plane error (assume in-plane is yz and out-plane is xz)
    errors.in_plane = vecnorm(nurbs(:, 2:3) - jig(:, 2:3), 2, 2);
    errors.out_plane = vecnorm(nurbs(:,[1, 3]) - jig(:, [1,3]), 2, 2);
    
end

    
% saving wrappers
function verbose_savefig(fig, file)
    savefig(fig, file);
    fprintf('Saved figure: %s\n', file);
    
end

function verbose_saveas(fig, file)
    saveas(fig, file);
    fprintf('Saved image: %s\n', file);
    
end

%% stereo_validation.m
%
% function to process a stereo data file
%
% - written by: Dimitri Lezcano


function [nurbs_interp_tf, jig_shape] = stereo_validation(file_num, curv_dir, kwargs)
    %% Argument Parsing
    arguments
       file_num
       curv_dir
       kwargs.save_dir = {}
    end

    %% File Preparation
    base_file = curv_dir + sprintf("left-right-%04d_", file_num) + "%s.%s";
%     nurbs_file = curv_dir + sprintf("left-right-%04d_nurbs-pts.txt", file_num);
    nurbs_file = sprintf(base_file, "nurbs-pts", "txt");
    
    % check if the files exist
    if exist(nurbs_file, 'file') ~= 2
        error("File does not exist: '%s'", nurbs_file);
        
    end
    
    %% Pull-in Data
    % nurbs-3D curve points
    nurbs_pts = readmatrix(nurbs_file);
    
    % determine the curvature
    re_patt = "k_([0-9]+.?[0-9]*)/";
    match = regexp(curv_dir, re_patt, 'tokens');
    
    if isempty(match)
        error("Could not find curvature regexp in directory: '%s'", curv_dir);
    end
    
    k = double(match{1})/1000; % curvature
    fprintf("Curvature found: %.2f 1/m\n", k*1000);
    
    %% Determine the shape of the Jig
    % preparation
    w = k * [1; 0; 0]; % curvature
    L = arclength(nurbs_pts); % arclength
    N = size(nurbs_pts, 1); % number of arclength points
    s = linspace(0, L, N); % arclength points
    
    % compute the jig_shape
    jig_shape = const_curv_shape(w, s)'; 
    
    %% Rigid body transform of nurbs points
    % standardize nurbs points ( for constant ds arclength points )
    nurbs_interp = interp_pts(nurbs_pts, s);
    
    % determine rigid body transform
    [R, p] = point_cloud_reg(nurbs_interp, jig_shape);
    
    % transform the nurbs points
    nurbs_interp_tf = (nurbs_interp * R') + p';
    
    %% Analysis
    errors = error_analysis(nurbs_interp_tf, jig_shape);
    
    %% Plotting
    % 3-D shape 
    fig_shape_3d = figure(1);
    set(fig_shape_3d,'units','normalized','position', [0, 0.4, 1/3, .5])
    plot3(jig_shape(:,3), jig_shape(:,1), jig_shape(:,2), 'g-', 'LineWidth', 2); hold on;
    plot3(nurbs_interp_tf(:,3), nurbs_interp_tf(:,1), nurbs_interp_tf(:,2), 'r-',...
        'LineWidth', 2); 
    hold off;
    legend('jig', '3D-Reconstruction'); 
    title(sprintf('3-D shapes: k = %.2f 1/m | img. # %04d', k*1000, file_num));
    xlabel('z','fontweight', 'bold'); ylabel('x','fontweight', 'bold'); 
    zlabel('y','fontweight', 'bold');
    axis equal; grid on;
    
    
    % 2-D shape
    fig_shape_2d = figure(2);
    set(fig_shape_2d,'units','normalized','position', [1/3, 0.4, 1/3, .5] )
    % - in-plane
    subplot(2,1,1);
    plot(jig_shape(:,3), jig_shape(:,2), 'g-', 'LineWidth', 2); hold on;
    plot(nurbs_interp_tf(:,3), nurbs_interp_tf(:,2), 'r-', 'LineWidth', 2);
    hold off;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('y [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    
    % - out-of-plane
    subplot(2,1,2);
    plot(jig_shape(:,3), jig_shape(:,1), 'g-', 'LineWidth', 2); hold on;
    plot(nurbs_interp_tf(:,3), nurbs_interp_tf(:,1), 'r-', 'LineWidth', 2);
    hold off;
    xlabel('z [mm]', 'FontWeight', 'bold'); ylabel('x [mm]', 'FontWeight', 'bold');
    axis equal; grid on;
    legend('jig', '3D-Reconstruction'); 
    
    sgtitle(sprintf('2-D shapes: k = %.2f 1/m | img. # %04d', k*1000, file_num));
    
    % error plots
    fig_err = figure(3);
    set(fig_err,'units','normalized','position', [2/3, 0.4, 1/3, .5])
    plot(s, 0.5 * ones(size(s)), 'r--', 'DisplayName', '0.5 mm'); hold on;
    plot(s, errors.L2, 'DisplayName', 'L2 Distance'); 
    plot(s, errors.dx, 'DisplayName', 'x-component');
    plot(s, errors.dy, 'DisplayName', 'y-component');
    plot(s, errors.dz, 'DisplayName', 'z-component');
    plot(s, errors.in_plane, 'DisplayName', 'in-plane');
    plot(s, errors.out_plane, 'DisplayName', 'out-of-plane');
    hold off;
    xlabel('s [mm]', 'fontweight', 'bold');
    ylabel('error [mm]','fontweight', 'bold');
    xlim([0, 1.1*max(s)]); ylim([0, max([1.1 * errors.L2', 1])]);
    legend(); grid on;
    title(sprintf('Errors: k = %.2f 1/m | img. # %04d', k*1000, file_num));
    
    
    %% Saving
    if ~isempty(kwargs.save_dir)
        % 3-D plot
        verbose_savefig(fig_shape_3d, sprintf(base_file, 'jig_shape-3d','fig'));
        verbose_saveas(fig_shape_3d, sprintf(base_file, 'jig_shape-3d','png'));
        
        % 2-D plot
        verbose_savefig(fig_shape_2d, sprintf(base_file, 'jig_shape-2d','fig'));
        verbose_saveas(fig_shape_2d, sprintf(base_file, 'jig_shape-2d','png'));
        
        % error plot
        verbose_savefig(fig_err, sprintf(base_file, 'jig_err','fig'));
        verbose_saveas(fig_err, sprintf(base_file, 'jig_err','png'));
        
    end
    
    

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

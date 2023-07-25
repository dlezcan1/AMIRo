function [pts_std, varargout] = interp_pts_col(pts, col, lu_query)
    arguments
        pts (:,3)
        col
        lu_query (:, 1)
    end
    lu_col = pts(:, col);

    lu_query( lu_query > max(lu_col) ) = max(lu_col); % cap the lookup arclength
    lu_query( lu_query < min(lu_col) ) = min(lu_col); % cap the lookup arclength
    
    % look-up for interpolation
    x_lu = pts(:,1); 
    y_lu = pts(:,2); 
    z_lu = pts(:,3);
    
    % interpolation
    x_interp = interp1(lu_col, x_lu, lu_query);
    y_interp = interp1(lu_col, y_lu, lu_query);
    z_interp = interp1(lu_col, z_lu, lu_query);
    
    % combine the output
    pts_std = [x_interp, y_interp, z_interp];
    varargout{1} = lu_query; % return the interpolation arclength just in case

end
%% readin_fbgparam_file
%
% function to read in the FBG param file and return kc and w_init
%
% - written by: Dimitri Lezcano

function [kc, w_init, L, theta0, tbl, varargout] = readin_fbgparam_file(filename)
    tbl = readtable(filename);
    
    tbl = mergevars(tbl, strcat("w_init_", string(1:3)), 'NewVariableName', 'w_init');
    
    if any(strcmp(tbl.Properties.VariableNames, 's_dbl_bend'))
        kc = tbl.kc;
        varargout{1} = tbl.s_dbl_bend(1);
    
    elseif any(strcmp(tbl.Properties.VariableNames, 'kc')) % 1 layer
        kc = tbl.kc;
        
    elseif any(strcmp(tbl.Properties.VariableNames, 'kc1')) && ... % 2 layer
           any(strcmp(tbl.Properties.VariableNames, 'kc2'))
       kc = [tbl.kc1, tbl.kc2];
       varargout{1} = tbl.s_crit;
    end
    
    w_init = reshape(tbl.w_init, [], 1);
    theta0 = tbl.theta0(1);
    L = tbl.L(1);
    
end
%% readin_fbgparam_file
%
% function to read in the FBG param file and return kc and w_init
%
% - written by: Dimitri Lezcano

function [kc, w_init, L, theta0] = readin_fbgparam_file(filename)
    tbl = readtable(filename);
    
    tbl = mergevars(tbl, "w_init_" + (1:3), 'NewVariableName', 'w_init');
    
    kc = tbl.kc(1);
    w_init = reshape(tbl.w_init, [], 1);
    theta0 = tbl.theta0(1);
    L = tbl.L(1);
    
end
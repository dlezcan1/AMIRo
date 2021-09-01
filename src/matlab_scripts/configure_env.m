%% configure_env.m
%
% script to configure environment for needle shape sensing using git
% directory
%
% Args:
%   - status: (on, off, toggle | default = toggle) whether to configure the environment
%   on or off. toggle will toggle the environment
%
% - written by: Dimitri Lezcano

%% Set Path
function configure_env(status)
    arguments
        status {mustBeMember(status, {'on', 'off', 'toggle'})} = 'toggle';
    end
    shapesensing_src = "../../shape-sensing/src/";
    shapesensing_src = what(shapesensing_src).path;
    
    pydir = fullfile('../');
    
    % check for toggle
    if strcmp(status, 'toggle')
        % check if shapesensing_src folder is on the path
        on = ismember(shapesensing_src, split(string(path),';'));
        
        if on
            status = 'off';
        else
            status = 'on';
        end
    end
    
    
    % configure the environment
    if strcmp(status, 'on')
        addpath(shapesensing_src);
        if count(py.sys.path, pydir) == 0
            insert(py.sys.path, int32(0), pydir);
        end
        disp("Shape sensing environment enabled.");
    else
        rmpath(shapesensing_src);
        disp("Shape sensing environment disabled.");
    end
end
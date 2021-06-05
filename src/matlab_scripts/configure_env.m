%% configure_env.m
%
% script to configure environment for needle shape sensing using git
% directory
%
% Args:
%   - status: (on, off | default = on) whether to configure the environment
%   on or off.
%
% - written by: Dimitri Lezcano

%% Set Path
function configure_env(status)
    arguments
        status {mustBeMember(status, {'on', 'off'})} = 'on';
    end
    shapesensing_src = "../../shape-sensing/src/";
    if strcmp(status, 'on')
        addpath(shapesensing_src);
    else
        rmpath(shapesensing_src);
    end
end
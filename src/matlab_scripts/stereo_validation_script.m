%% stereo_validation_scipt.m
%
% script to process all of the files in the data directory for stereo 3-D
% reconstruction of the needle shape
%
% - written by: Dimitri Lezcano

%% Set-up
data_dir = "../../Test_Images/stereo_needle/stereo_validation_jig/";
curvature_dirs = data_dir + ls(data_dir + "k_*") + "/"; 

% regex set-up
re_patt = "left-right-([0-9]{4})";


%% Process each curvature directory
for i = 1:length(curvature_dirs)
    curv_dir = curvature_dirs(i);
    fprintf("Processing directory: %s\n", curv_dir);
    disp("Gathering image file numbers...");
    img_nums = get_stereo_numbers(curv_dir);
    disp(img_nums);
    
    for img_num = img_nums
        stereo_validation(img_num, curv_dir, 'save_dir', curv_dir);
        pause(1);
    end  

    fprintf('Completed.\n\n' );
    
end

%% Functions
function img_nums = get_stereo_numbers(curv_dir)
    % regex setup
    re_patt = "left-right-([0-9]{4})";
    
    % grab files
    nurb_files = dir(curv_dir + "left-right-*_nurbs-pts.txt");
    
    % find the image numbers
    img_nums = [];
    for i = 1:length(nurb_files)
        match = regexp(nurb_files(i).name, re_patt, 'tokens');
        if ~isempty(match)
            num = str2num(match{1}{1});
            img_nums = [img_nums, num];
            
        end
    end
end
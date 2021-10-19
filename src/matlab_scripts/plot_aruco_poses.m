%% plot_aruco_poses.m
%
% plot the aruco poses for an insertion experiment

%% Set-Up
% directories to iterate throughn ( the inidividual trials )
expmt_dir = "../../data/3CH-4AA-0004/2021-09-29_Insertion-Expmt-1/";
trial_dirs = dir(fullfile(expmt_dir, "Insertion*/"));
mask = strcmp({trial_dirs.name},".") | strcmp({trial_dirs.name}, "..") | strcmp({trial_dirs.name}, "0");
trial_dirs = trial_dirs(~mask); % remove "." and ".." directories and "0" directory
trial_dirs = trial_dirs([trial_dirs.isdir]); % make sure all are directories

% ARUCO pose file
tr_aruco_pose_file = "left-right_aruco-poses";

%% Plot the directories
ins_hole = -1;
counter = 0;
fig_pose = figure(1);

for i = 1:numel(trial_dirs)
   % experimental params
   L        = str2double(trial_dirs(i).name);
   re_ret   = regexp(trial_dirs(i).folder, "Insertion([0-9]+)", 'tokens');
   hole_num = str2double(re_ret{1}{1});
   
   % read the pose
   aruco_file = fullfile(trial_dirs(i).folder, trial_dirs(i).name, tr_aruco_pose_file);
   [l_pose, r_pose] = read_aruco_pose(aruco_file);
   
   if ins_hole ~= hole_num
       if ins_hole > 0
           fprintf("Insertion%d # Poses: %d\n", ins_hole, counter); 
           disp("Press enter to continue...");
           pause;
       end
       hold off;
       counter = 1;
       ins_hole = hole_num;
   else
       counter = counter + 1;
   end
   plotf(l_pose); hold on;
   title(sprintf("Insertion %d", ins_hole));
   
    
end
fprintf("Insertion%d # Poses: %d\n", ins_hole, counter); 
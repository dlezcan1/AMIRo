%% wl_shiftify.m
%
% scipt to take fbgdata file and shift the wavelengths using a reference
% wavelength
%
% - wrriten by: Dimitri Lezcano

%% Script Set-up
save_bool = true;

%% File set-up
% reference wavelength
ref_dir = "./";
ref_file = ref_dir + "Reference_wavelength.xls";

% FBGdata file list
fbgdata_files = dir("Insertion*/*/FBGdata.xls");
fbgdata_shifts_file = "FBGdata_shifts.xls";
fbgdata_meanshift_file = "FBGdata_meanshift.xls";

% directory separator
if ispc
    dir_sep = '\';
    
else
    dir_sep = '/';
end

%% Read in the reference wavelengths
ref_mat = readmatrix(ref_file);
ref_mat = ref_mat(all(ref_mat > 0, 2), :); % remove 0 rows
ref_wls = mean(ref_mat, 1);

%% Iterate through FBGData files and create the reference matrices
for i = 1:length(fbgdata_files)
    % unpack file name structure
    d = fbgdata_files(i).folder;
    fbg_file = strcat(d, dir_sep, fbgdata_files(i).name);
    
    % read in the fbgdata
    fbg_mat = readmatrix(fbg_file);
    fbg_mat = fbg_mat(all(ref_mat > 0, 2), :);
    
    % shift the wavelengths
    fbg_shift_mat = fbg_mat - ref_wls;
    
    % mean the shifts
    fbg_shift_mean = mean(fbg_shift_mat, 1);
    
    % save (if turned on)
    if save_bool
        writematrix(fbg_shift_mat, strcat(d, dir_sep, fbgdata_shifts_file));
        fprintf('Saved file: %s\n', strcat(d, dir_sep, fbgdata_shifts_file));
        
        writematrix(fbg_shift_mean, strcat(d, dir_sep, fbgdata_meanshift_file));
        fprintf('Saved file: %s\n', strcat(d, dir_sep, fbgdata_meanshift_file));
        disp(' ');
        
    end
end

%% dicomdir2json
%
% Author: Dimitri Lezcano
%
% This is a function to convert the dicomdir data into a human-readable 
% json file

%% function
function [dcm_info] = dicomdir2json(dicomdir_file, output_file)
    %% argument parsing
    arguments
        dicomdir_file string
        output_file string = ""
    end
    if strcmp(output_file, "")
        output_file = strcat(dicomdir_file, ".json");
    end

    %% Load the dicomdir file
    dcm_info = images.dicom.parseDICOMDIR(dicomdir_file);

    fid = fopen(output_file, "w");
    
    fprintf(fid, "%s", jsonencode(dcm_info));

    fclose(fid);
    fprintf("Wrote json DICOMDIR file to: %s\n", output_file);
        




end

function parse_study(study)
    arguments
        study struct;
    end

    studydate = datetime(...
        strcat(...
            study.Payload.StudyDate,...
            "-",...
            study.Payload.StudyTime...
        ),...
        "InputFormat", "yyyyMMdd-HHmmss.S" ...
    );

end

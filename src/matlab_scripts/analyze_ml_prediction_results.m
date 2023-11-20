% analyze_ml_prediction_results.m


prediction_dir = "../../prediction_ml";
results_dir    = fullfile(prediction_dir, "results");

error_file   = "Prediction_Errors.xlsx";
results_files = dir(fullfile(results_dir, "*.xlsx-plots", error_file));
results_files = results_files(...
    isfile(fullfile({results_files.folder}, {results_files.name}))...
);



%% Process each model results
cumulative_table = [];
for i = 1:numel(results_files)
    % read in the results
    tbl = readtable( ...
        fullfile(results_files(i).folder, results_files(i).name),...
        'TextType', 'string'...
    );

    folder_psplit = pathsplit(results_files(i).folder);
    model_name = strrep( ...
        folder_psplit{end}, ...
        ".xlsx-plots", ...
        ""...
    );

    tbl.Model(:) = model_name;

    if i == 1
        cumulative_table = tbl;
    else
        cumulative_table = [cumulative_table; tbl];
    end

end

%% Save the table
writetable( ...
    cumulative_table,...
    fullfile(results_dir, 'cumulative_prediction_errors.xlsx'), ...
    'WriteRowNames', false...
);
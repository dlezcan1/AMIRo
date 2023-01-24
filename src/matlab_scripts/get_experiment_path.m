function path = get_experiment_path( ...
    data_dir, experiment, insertion_hole, insertion_depth...
)
arguments
    data_dir        string;
    experiment      string;
    insertion_hole  {mustBeInteger} = 0;
    insertion_depth {mustBeNumeric} = -1;
end
    
%get_experiment_path Generates the path to a specific experiment file

    path = fullfile(data_dir, experiment);
    
    if insertion_hole <= 0
        return 
    end
    path = fullfile(path, fprintf("Insertion%d", insertion_hole));
    
    if insertion_depth < 0
        return
    end
    path = fullfile(path, num2str(insertion_depth));
    
end


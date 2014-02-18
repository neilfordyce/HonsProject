function F = read_feature(feature_dir, i)
    feature_file = fullfile(feature_dir, sprintf('F%d.mat', i));
    load(feature_file, 'F');
    
    %Filter to only use the most important features
    load('maxIndex.mat', 'maxIndex');
    F = F(:,maxIndex > 0);
end
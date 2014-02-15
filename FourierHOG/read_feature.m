function F = read_feature(feature_dir, i)
    feature_file = fullfile(feature_dir, sprintf('F%d.mat', i));
    load(feature_file, 'F');
    %F = [F(:,105:115), F(:,end)];
end
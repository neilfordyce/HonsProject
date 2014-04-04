function [ins] = load_data(data_dir, gt_mask_dir, image_indices)

% Directories
imagedir   = [data_dir, '/golgi'];
gt_dir     = [data_dir, gt_mask_dir];

% Files
D = dir([imagedir, '/*jpg']);
if(~exist('image_indices')), image_indices=1:length(D); end;

M = length(image_indices);
for m = 1:M
    
    id = image_indices(m);
    
    fprintf('Loading ', m);
    % Image
    image_filename = sprintf('%s/%s', imagedir, D(id).name);
    [d ins.name{m} e] = fileparts(image_filename);
    ins.image_filename{m} = image_filename;
    fprintf('\tIMAGE: %s\n', image_filename);
    I = imread(image_filename);
    ins.image_size{m} = size(I);
    
    % GROUNDTRUTH
    ins.gt_filename{m} = sprintf('%s/%s.jpg', gt_dir, ins.name{m});
end

end

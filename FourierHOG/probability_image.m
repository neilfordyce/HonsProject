%%AUTHOR Neil Fordyce
%%DATE   28/01/14
function [ img ] = probability_image( data, tas_params, output_path )
%Sequentially passes all the data into showboxes() to mark the detetections
%and output to file if output_path is specified
    for i=1:numel(data.image_filename)
        img = imread(data.image_filename{i});
        img = rgb2gray(img);
        img = imresize(img, tas_params.scale);
        img = -inf * ones(size(img));
        
        for j=1:numel(data.score{i})
            det_bbox = data.dets{i}(j,:);
            det_center = round(0.5 * (det_bbox(3:4)+det_bbox(1:2)));    %find bbox center
            img(det_center(2), det_center(1)) = data.score{i}(j);
        end
        
        if ~isempty(output_path)
            full_output_path = fullfile(output_path, data.name{i});
            showboxes(img, dets, det_count, full_output_path);
        else
            imshow(img, [])
            pause(0.5)
        end 
    end 
end

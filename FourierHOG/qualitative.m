%%AUTHOR Neil Fordyce
%%DATE   28/01/14
function [ ] = qualitative( data, params, output_path )
%Sequentially passes all the data into showboxes() to mark the detetections
%and output to file if output_path is specified
    for i=1:numel(data.image_filename)
        img = imread(data.image_filename{i});
        img = imresize(img, params.scale);
        
        scores = data.score{i};
        det_count = numel(scores(scores > params.cand_threshold));

        dets = data.dets{i};
        dets = dets';
        dets = dets(:);
        dets = dets';

        if ~isempty(output_path)
            full_output_path = fullfile(output_path, data.name{i});
            showboxes(img, dets, det_count, full_output_path);
        else
            showboxes(img, dets, det_count);
        end 
    end 
end

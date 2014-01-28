function [ ] = qualitative( data, tas_params, i )
%QUALITATIVE Summary of this function goes here
%   Detailed explanation goes here
    scores = data.score{i};
    det_count = numel(scores(scores>tas_params.cand_threshold));
    
    img = imread(data.image_filename{i});
    img = imresize(img, tas_params.scale);
    
    dets = data.dets{i};
    dets = dets';
    dets = dets(:);
    dets = dets';
    
    showboxes(img, dets, det_count);
end


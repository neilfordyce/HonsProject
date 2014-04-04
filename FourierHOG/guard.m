%AUTHOR Neil Fordyce
function [ I ] = guard( I )
%% Crops the image to remove edge regions with detections affected by padding

load_params
padding = param.featureScale * 5; % Zero padding size in FourierHOG.m

            %top,   bottom, left,   right
crop_box = [padding,padding,padding,80]; % 80 is enough to crop off the calibration bar from all the images

I = I(crop_box(1):end-crop_box(2), crop_box(3):end-crop_box(4));

end


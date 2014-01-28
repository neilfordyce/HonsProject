%%AUTHOR Neil Fordyce
%%DATE   28/01/14
function [ mask ] = read_mask( mask_filename, scale )
%Read in all the mask files and set matrix values which indicate the ground
%truth of each pixel to the rest of the system
    mask = imread(mask_filename);
    mask = rgb2gray(mask);
    mask(mask > 0 & mask < 255) = 1;  % all gray areas to 1
    mask(mask == 255) = 2;  % all white areas to 2
    mask = imresize(mask, scale);
end


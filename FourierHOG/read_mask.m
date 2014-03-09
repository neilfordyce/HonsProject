%%AUTHOR Neil Fordyce
%%DATE   28/01/14
function [ mask ] = read_mask( mask_filename, scale )
%Read in all the mask files and set matrix values which indicate the ground
%truth of each pixel to the rest of the system
    mask = imread(mask_filename);
    mask = rgb2gray(mask);
    mask(mask <= 5) = 0;
    
    %Set gray areas = 2 => ambiguous regions = positive regions
    %Set gray areas = 1 => ambiguous regions != positive regions
    mask(mask > 5 & mask < 255) = 1; % all gray areas to 1
    mask(mask == 255) = 1;  % all white areas to 2
    mask = imresize(mask, scale);
    
    %iterpolation is messsing up the mask edges, so set to one first then fix it now
    mask = mask *2;
end

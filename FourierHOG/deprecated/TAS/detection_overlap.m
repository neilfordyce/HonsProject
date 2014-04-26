%%AUTHOR Neil Fordyce
%%DATE   27/01/14
%Rewrite of TAS detection_overlap.m to support own labeling framework, for
%non-rigid objects.  Orignal TAS detection overlap compared bounding boxes,
%which are not be suitable for Golgi groundtruths.
function ov = detection_overlap(bb, mask)
    %Crop the mask to the bounding box
    bb_crop = mask(bb(2):bb(4), bb(1):bb(3));
    total_pixels = numel(bb_crop);
    golgi_pixels = numel(bb_crop(bb_crop == 2));
    ov = golgi_pixels / total_pixels;
end
  

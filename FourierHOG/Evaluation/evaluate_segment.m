%%Neil Fordyce
function [ accuracy, F1, missed_seg, false_seg ] = evaluate_segment( gt, seg )
%Evaluates the quality of a segmentation

%gt 0=neg, 1=ambiguous, 2=pos
%seg is a binary image
%so in eval_image, we have; 0=TN, 2=FN, 3=FP, 5=TP, 1=4=N/A
missed_seg=0; false_seg=0;
gt=double(gt);
%seg(gt==1) = 0;
eval_image = gt + (3*seg);

FN = numel(eval_image(eval_image==2));
FP = numel(eval_image(eval_image==3));
TP = numel(eval_image(eval_image==5));
F1 = (2*TP) / ((2*TP)+FP+FN);

%Now evalate each of the segmentations seperately
CC = bwconncomp(eval_image);
PixelIdxList = CC.PixelIdxList;
for i=1:numel(PixelIdxList)    
    eval_image_connected_comp = eval_image(PixelIdxList{i});
   
  %  if any(ismember([4 1], eval_image_connected_comp))
   %     continue;
   % end
    
    %TN = numel(eval_image(eval_image==0));
    FN = numel(eval_image_connected_comp(eval_image_connected_comp==2));
    FP = numel(eval_image_connected_comp(eval_image_connected_comp==3));
    TP = numel(eval_image_connected_comp(eval_image_connected_comp==5));
    accuracy(i) = TP / (TP+FP+FN);
    
    if isnan(accuracy(i)) %numel(eval_image_connected_comp(eval_image_connected_comp==1)) > 0 | numel(eval_image_connected_comp(eval_image_connected_comp==4)) > 0
        continue;   %Don't count missed ambiguous regions
    end
    
    %Count false detections and missed detections, if any
    if accuracy(i) == 0
        if any(ismember([0,4], eval_image_connected_comp))
            continue;   %Don't count missed ambiguous regions
        end
        
        if FN > 0
            missed_seg = missed_seg + 1;
        elseif FP > 0
            false_seg = false_seg + 1;        
        end
    end
end

%{

%Take each golgi region in the GT in turn, 
%find out how much of it was found in the labeling L
    CC_seg = bwconncomp(seg);
    CC = bwconncomp(gt);
    %STATS = regionprops(CC, 'Area');
    seg_PixelIdxList = CC_seg.PixelIdxList;
    gt_PixelIdxList = CC.PixelIdxList;
    
    TP = [];
    for i=1:numel(gt_PixelIdxList)
        for j=1:numel()
       TP(i) = sum(L(gt_PixelIdxList{i}));
       TP(i) = TP(i) / numel(gt_PixelIdxList{i});
    end
%}
end

function [ accuracy, F1 TP_seg, FP_seg, FN_seg ] = evaluate_segment( gt, seg, thresh )
%Evaluates the quality of a segmentation

%gt and seg are binary images
%so in eval_image, we have; 0=TN, 1=FN, 2=FP, 3=TP
TP_seg=0; FP_seg=0; FN_seg=0;
eval_image = im2bw(gt, 0) + (2*seg);

FN = numel(eval_image(eval_image==1));
FP = numel(eval_image(eval_image==2));
TP = numel(eval_image(eval_image==3));
F1 = (2*TP) / ((2*TP)+FP+FN);

%Now evalate each of the segmentations seperately
CC = bwconncomp(eval_image);
PixelIdxList = CC.PixelIdxList;
for i=1:numel(PixelIdxList)    
    eval_image_connected_comp = eval_image(PixelIdxList{i});
    %TN = numel(eval_image(eval_image==0));
    FN = numel(eval_image_connected_comp(eval_image_connected_comp==1));
    FP = numel(eval_image_connected_comp(eval_image_connected_comp==2));
    TP = numel(eval_image_connected_comp(eval_image_connected_comp==3));
    accuracy(i) = TP / (TP+FP+FN);
    
    if accuracy(i) < thresh
        if FP > FN
            FP_seg = FP_seg + 1;
        else
            FN_seg = FN_seg + 1;
        end
    else
        TP_seg = TP_seg + 1;
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

function [ error ] = evaluate_segment( gt, seg )
%Evaluates the quality of a segmentation
TP = numel(seg(gt(gt==1) & seg==1));
FP = numel(seg(gt(gt==0) & seg==1));
FN = numel(seg(gt(gt==1) & seg==0));

error = TP / (TP+FP+FN);
end

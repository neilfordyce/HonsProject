%AUTHOR Neil Fordyce
load_seg_params

for i=2:0.5:5
    seg_param.gradient_smoothing_sigma = i;
    segment(seg_param, seg_param.gradient_smoothing_sigma, 'Sigma');
end

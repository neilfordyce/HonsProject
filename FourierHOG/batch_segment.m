%AUTHOR Neil Fordyce
load_seg_params

seg_param.svm_scores;

det_dir = {'FourierHOG_Prob73569998629', 'FourierHOG_Prob73570003251'};

for i=1:numel(det_dir)
    seg_param.svm_scores = ['C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\variable_scale\', det_dir{i}, '\'];
    load([seg_param.svm_scores, '\data'], 'data');
    legend_text = data.legend_text;
    segment_feeder(seg_param, legend_text(regexp(legend_text, '\d')), 'Detection');
end

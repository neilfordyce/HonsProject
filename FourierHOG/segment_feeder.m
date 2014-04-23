% Author Neil Fordyce
% Feeds all the images to segment() to be segemented then runs evaluators
% to quantify performance
function [performance]=segment_feeder(seg_param, x_label, batch_variable)%data)
load_params;

if not(exist('seg_param','var'))
    load_seg_params     %seg params
end

if not(exist('batch_variable','var'))
    batch_variable = '';
end
    
performance = {};  performance.false_seg = 0; performance.missed_seg = 0;
if exist('x_label', 'var')
   performance.x_label = x_label; 
end

accuracy = [];

gt_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\output\gt_masks_2';
em_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\';
prob_dir = seg_param.svm_scores;
output_dir = fullfile('C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\segment', batch_variable, ['segment', num2str(round(now*100000))]);
mkdir(output_dir);
prob_files = dir([prob_dir, '*jpg']);

file_count = length(prob_files);
%file_count = 3;

for file_i = 1:file_count
%for file_i = 14:14
    %% read the SVM image, electron micrograph image and ground truth for evaluation  
    filename = prob_files(file_i).name;
    im = imread(fullfile(prob_dir, filename));
    %im = data.score{file_i};
    em_im = imread(fullfile(em_dir, filename));

    %im = slic_segment(im, em_im);

    im = im2double(im);

    em_im = rgb2gray(em_im);
    em_im = imresize(em_im, param.scale);
    %em_im = im2double(em_im);
    
    gt = read_mask( fullfile(gt_dir, filename), param.scale);
    gt=guard(gt);
    
    [em_im, L] = segmenter(im, em_im, seg_param);
    
    %% Do evaluation
    [acc, ~, missed_seg, false_seg] = evaluate_segment(gt, L);
    accuracy = [accuracy, acc];
    
    performance.false_seg = performance.false_seg + false_seg;
    performance.missed_seg = performance.missed_seg + missed_seg;
    
    %% Add evaluation scores to bottom
    out_height = 40;
    text_em_im = [em_im; zeros(out_height, size(em_im, 2), size(em_im,3))];
    
    htxtins = vision.TextInserter( sprintf('%#1.4f      ', acc));
    htxtins.Color = [255,255,255]; % [red, green, blue]
    htxtins.FontSize = 24;
    htxtins.Location = [ 1, size(em_im, 1) + out_height/2]; % [x y]
    htxtins.Antialiasing = true;
    htxtins.Font = 'Consolas';
    text_em_im = step(htxtins, text_em_im);
    
    imshow(text_em_im, []);
    imwrite(text_em_im, fullfile(output_dir, filename));

end

performance.accuracy = accuracy;
performance.acc_above_zero = numel(accuracy(accuracy>0));
performance.acc_above_thresh = numel(accuracy(accuracy>0.5));
performance.mean_ji = mean(accuracy(accuracy>0));

save([output_dir, '\performance'], 'performance');
save([output_dir, '\seg_param'], 'seg_param');

end

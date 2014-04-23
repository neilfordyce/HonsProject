%% Load params and SVM model
load_params
load('model.mat', 'model');

%% Load image
I=imread('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\110511C_2_IPL.jpg');
I=rgb2gray(I);
I=im2double(I);
I=imresize(I, param.scale);

subplot(1,3,1);
imshow(I);

%% Local Detection
%Feature extraction
prepareKernel(param.featureScale, param.maxOrder, 0);
tic
    F=FourierHOG(I, param.featureScale, 0:param.maxOrder);
toc

%Detect using trained SVM model
Y_hat = F * model.w(1:end-1)' + model.w(end);
Y_hat = reshape(Y_hat, size(I));

subplot(1,3,2);
imshow(Y_hat, []);

%% Segmentation
tic
    [seg_im, labelling] = segmenter(Y_hat, I);
toc

subplot(1,3,3);
imshow(seg_im, []);

%% Evaluate segmentation
gt = read_mask('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\output\gt_masks_2\110511C_2_IPL.jpg', param.scale);
gt = guard(gt);
[acc, ~, missed_seg, false_seg] = evaluate_segment(gt, labelling)

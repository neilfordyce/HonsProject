%% Try the code on a computing server.
% It will compue and store the feautres pixel-wisely for 30 792 * 636 pixel
% images, so it requires 30g memory to run.
clearvars
close all
dbstop if error

% parameters
param.featureScale = 6;
param.NMS_OV  = 0;  %non-max. suppression
param.bbRadius = 30;
param.indifferenceRadius = param.bbRadius;
param.sample_count = 30;
param.neg_sample_count = 3000;  %Per image
param.pos_sample_multiplier = 50000;

initrand();

% requires export_fig from http://www.mathworks.com/matlabcentral/fileexchange/23629-exportfig
addpath('/home/liu/Matlab/export_fig');
% requires liblinear
addpath(genpath('/home/liu/Matlab/liblinear-1.8'));

% requires the TAS package (and image data) from http://ai.stanford.edu/~gaheitz/Research/TAS/
% download and unfold it into the current path
% then add the path to the TAS package
addpath(genpath('/home/liu/Dropbox/doc/IJCV/code/FourierHOG/TAS'));

%% load data and TAS setting
search  %training tas_params
data = load_data('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images', 1:param.sample_count);
rands = randperm(length(data.image_filename));

%Make dir to output certainty images
Y_hat_dir = fullfile('C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs', ['FourierHOG_Prob', num2str(round(now*100000))]);
mkdir(Y_hat_dir);

feature_dir = 'C:\Users\Neil\golgi_fourier_features';

%%Read images and ground truth masks
Image = [];
for m = 1:length(data.image_filename)
    Image{m} = im2double(rgb2gray(imread(data.image_filename{m})));
    Image{m} = imresize(Image{m}, tas_params.scale);
    data.mask{m} = read_mask(data.gt_filename{m}, tas_params.scale);
end

%% computing features
N = numel(Image);
F = cell(1, N);
prepareKernel(param.featureScale, 4, 0);
pause(0.1);
for i = 1:length(data.image_filename)
    feature_file = fullfile(feature_dir, sprintf('F%d.mat', i));
    %Create and compute a feature file if one doesn't already exist
    if exist(feature_file) ~= 2
        disp(['extracting features from image' num2str(i)]);
        tic
        F = FourierHOG(Image{i},param.featureScale);
        save(feature_file, 'F');
        toc;
    end
end

fprintf('Feature Dimension: %d \n' ,size(F,2) );

data.score = cell(1,N);

for ifold = 1:5
    sample_mask = false(1,N);
    sample_mask( (ifold - 1) * N/5 + (1:N/5) ) = true;
    
    trainIndex = rands(~sample_mask)
    testIndex = rands(sample_mask)
    
    % get training samples:  792 * 636 pixels per image
    Pos = [];
    Neg = [];
    % get mask and data
    for i = trainIndex
        F = read_feature(feature_dir, i);

        %% downsampling
        %number of pos samples is defined by the number available to take
        total_pixels = numel(data.mask{i}(:));
        pos_pixels = numel(data.mask{i}(data.mask{i}==2));
        pos_sample_count = round((pos_pixels / total_pixels) * param.pos_sample_multiplier)
        
        index1 = find(data.mask{i}(:) == 2);
        index2 = randsample(numel(index1), pos_sample_count);   % draw positive samples from image
        Pos{i} = F(index1(index2), :);
        
        index1 = find(data.mask{i}(:) == 0);
        index2 = randsample(numel(index1), param.neg_sample_count);   % draw 3000 negative samples per image
        Neg{i} = F(index1(index2), :);
    end
    Pos1 = double(cell2mat(Pos(trainIndex)'));
    Neg1 = double(cell2mat(Neg(trainIndex)'));
    
    %%
    fprintf('train linear SVM\n');
    tic
    % simple normalization
    ABSMAX = max([Pos1;Neg1]) + eps;
    trainX = sparse(bsxfun(@times, [Pos1;Neg1], 1./ ABSMAX ));
    
    model = train([ones(size(Pos1,1),1); ones(size(Neg1,1),1) * 0], trainX, '-B 1 -c 1');
    
    model.w(1:end-1) = model.w(1:end-1) ./ ABSMAX;
    toc
    %% hard sample mining and retrain
    fprintf('hard sample mining\n');
    tic
    HardNeg = cell(1,length(data.image_filename));
    for i = 1:length(data.image_filename)
        if(~ ismember(i,trainIndex))
            continue;
        end
        F = read_feature(feature_dir, i);

        %% detection
        votes =  F * model.w(1:end-1)' + model.w(end);
        Y_hat = reshape(votes, [size(Image{1}, 1), size(Image{1}, 2)]);
        %% sample negative samples based on the distances to the hyperplane
        Y_hat = max(Y_hat + 1, 0) .* (data.mask{i} == 0);
        index = unique(randsample(numel(Y_hat), 3000, true, Y_hat(:)));
        HardNeg{i} = F(index, :);
    end
    HardNegM = double(cell2mat(HardNeg(trainIndex)'));
    toc
    %% train and predict
    fprintf('train linear SVM\n');
    tic
    trainX = sparse(bsxfun(@times, [Pos1;HardNegM], 1./ ABSMAX ));
    size(trainX)
    model = train([ones(size(Pos1,1),1); zeros(size(HardNegM,1),1) ], trainX, '-B 1 -c 1');
    model.w(1:end-1) = model.w(1:end-1) ./ ABSMAX;
    save('model.mat', 'model');
    toc
    
    %% detection
    fprintf('detection\n');
    tic
    for i = 1:length(data.image_filename)
        if(~ ismember(i,testIndex))
            continue;
        end
        F = read_feature(feature_dir, i);
        votes = F * model.w(1:end-1)' + model.w(end);
        Y_hat = reshape(votes, [size(Image{1}, 1), size(Image{1}, 2)]);
        
        scale_Y_hat = step(vision.ContrastAdjuster, Y_hat); %Contrast scale the certainties image
        imwrite(scale_Y_hat, fullfile(Y_hat_dir, [data.name{i} '.jpg'])); %Store the certainty image

        data.score{i} = Y_hat;
    end
    % results are accumulated in the cross-validation process
end
%% evaluate 
[data.dist, data.performance_score] = evaluate_quantative( data );
plot_evaluation(data);

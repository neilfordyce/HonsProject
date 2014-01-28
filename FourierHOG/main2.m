%% Try the code on a computing server.
% It will compue and store the feautres pixel-wisely for 30 792 * 636 pixel
% images, so it requires 30g memory to run.
clearvars
close all
dbstop if error



% parameters
param.featureScale = 6;
param.NMS_OV  = 0.5;  %non-max. suppression
param.bbRadius = 40;
param.indifferenceRadius = param.bbRadius;

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
%run('TAS/example/Params/search.m'); 
search  %training tas_params
data = load_data('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images', 1:10);
rands = randperm(length(data.image_filename));

%%
I = im2double(imread(data.image_filename{1}));
figure('Name', 'example image');
imshow(I);shg;
pause(0.5);

%%
Image = [];
scale = 0.2;
for m = 1:length(data.image_filename)
    Image{m} = im2double(imread(data.image_filename{m}));
    Image{m} = imresize(Image{m}, tas_params.scale);
    mask{m} = read_mask(data.gt_filename{m}, tas_params.scale);
end

%% computing features
N = numel(Image);
F = cell(1, N);
prepareKernel(param.featureScale, 4, 0);
pause(0.1);
for i = 1:length(data.image_filename)
    disp(['extracting features from image' num2str(i)]);
    tic
    F{i} = FourierHOG(Image{i},param.featureScale);
    toc;
end

fprintf('Feature Dimension: %d \n' ,size(F{i},2) );

data.dets = cell(1,N);
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
        % indifferent positions
        se = strel('disk',  param.indifferenceRadius, 0);
        dmask = imdilate(mask{i}, se);
        % add positive samples around the centers
        se = [1 1 1;1 1 1;1 1 1];
        mask{i} = imdilate(mask{i}, se);
        
        mask{i}( xor(dmask, mask{i}) ) = 0.5;
        %% downsampling
        %number of pos samples is defined by the number available to take
        total_pixels = numel(mask{i}(:));
        pos_pixels = numel(mask{i}(mask{i}==2));
        pos_sample_count = (pos_pixels / total_pixels) * 10000
        
        index1 = find(mask{i}(:) == 2);
        index2 = randsample(numel(index1), pos_sample_count);   % draw positive samples from image
        Pos{i} = F{i}(index1(index2), :);
        
        index1 = find(mask{i}(:) == 0);
        index2 = randsample(numel(index1), 3000);   % draw 3000 negative samples per image
        Neg{i} = F{i}(index1(index2), :);
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
        se = strel('disk',  param.indifferenceRadius   ,0);
        dmask = imdilate(mask{i}, se);
        se = [1 1 1;1 1 1;1 1 1];
        mask{i} = imdilate(mask{i}, se);
        mask{i}( xor(dmask, mask{i}) ) = 0.5;
        %% detection
        votes =  F{i} * model.w(1:end-1)' + model.w(end);
        Y_hat = reshape(votes, [size(Image{1}, 1), size(Image{1}, 2)]);
        %% sample negative samples based on the distances to the hyperplane
        Y_hat = max(Y_hat + 1, 0) .* (mask{i} == 0);
        index = unique(randsample(numel(Y_hat), 3000, true, Y_hat(:)));
        HardNeg{i} = F{i}(index, :);
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
        votes = F{i} * model.w(1:end-1)' + model.w(end);
        Y_hat = reshape(votes, [size(Image{1}, 1), size(Image{1}, 2)]);
        %         figure(98);clf;
        %         imagesc(Y_hat); axis equal tight off;
        %         export_fig([data.image_filename{i}(1:end-4) '_det.png']);pause(0.5);
        %         save([data.image_filename{i}(1:end-4) '_det.mat'], 'Y_hat');
        %%
        bw = imregionalmax(Y_hat);
        dt = [];
        STATS = regionprops(bw, 'Centroid');
        for j = 1:numel(STATS)
            cc = round(STATS(j).Centroid);
            v = Y_hat(cc(2),cc(1));
            dt = [dt; [cc(1) - param.bbRadius , cc(2) - param.bbRadius , cc(1) + param.bbRadius , cc(2) + param.bbRadius , v]];
        end
        pick = nms(dt,param.NMS_OV);
        dt = dt(pick,:);
        I = Image{i};
        dt = clipboxes(I, dt);
        data.dets{i} = dt(:,1:4);
        data.score{i} = dt(:,5);
    end
    % results are accumulated in the cross-validation process
end
%% using the evaluation tool from TAS package
pr = det_rpc(data,data.score,tas_params.truth_threshold, 'r-'); pause(0.5);
pr.ap
figure('Name', 'PR'); plot(pr.recall, pr.precision, 'r-'); shg; pause(0.5)

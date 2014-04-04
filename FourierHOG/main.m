%%TODO Clean up here
%% Try the code on a computing server.
% It will compue and store the feautres pixel-wisely for 30 792 * 636 pixel
% images, so it requires 30g memory to run.
if not(exist('batch_run','var'))
    %Clear the vars unless batch mode indicated
    clearvars
    close all
    dbstop if error 
end

if not(exist('param','var'))
    load_params  %training params
end

if not(exist('output_dir','var'))
    output_dir = '';
end

initrand();

%% load data
data = load_data('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images', param.gt_mask_dir,  1:param.sample_count);
rands = randperm(length(data.image_filename));

%Add roc plot legend text if it is there
if exist('legend_text','var')
   data.legend_text = legend_text; 
end

%Make dir to output certainty images
Y_hat_dir = fullfile('C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs', output_dir, ['FourierHOG_Prob', num2str(round(now*100000))]);
mkdir(Y_hat_dir);

feature_dir = 'C:\Users\Neil\golgi_fourier_features_gt2';

%%Read images and ground truth masks
Image = [];
for m = 1:length(data.image_filename)
    Image{m} = im2double(rgb2gray(imread(data.image_filename{m})));
    Image{m} = imresize(Image{m}, param.scale);
    data.mask{m} = read_mask(data.gt_filename{m}, param.scale);
end

%% computing features
N = numel(Image);
F = cell(1, N);
prepareKernel(param.featureScale, param.maxOrder, 0);
pause(0.1);
for i = 1:length(data.image_filename)
    feature_file = fullfile(feature_dir, sprintf('F%d.mat', i));
    %Create and compute a feature file if one doesn't already exist
    if exist(feature_file) ~= 2
        disp(['extracting features from image' num2str(i)]);
        tic
        F = FourierHOG(Image{i}, param.featureScale, 0:param.maxOrder);
        save(feature_file, 'F', '-v7.3');
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

        for j=1 : size(F, 2)
            F(:,j) = step(vision.ContrastAdjuster('OutputRangeSource', 'Property', 'OutputRange', [0,1]), F(:,j));
        end
        
        %% random sampling 
        total_pixels = numel(data.mask{i}(:));
        pos_pixels = numel(data.mask{i}(data.mask{i}==2));
        pos_sample_count = round((pos_pixels / total_pixels) * param.pos_sample_multiplier);
        
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
    %model = svmtrain(trainX, [ones(size(Pos1,1),1); ones(size(Neg1,1),1) * 0]);
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
        
        for j=1 : size(F, 2)
            F(:,j) = step(vision.ContrastAdjuster('OutputRangeSource', 'Property', 'OutputRange', [0,1]), F(:,j));
        end

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
    %model = svmtrain(trainX, [ones(size(Pos1,1),1); ones(size(Neg1,1),1) * 0]);
    %model_class = classRF_train(trainX,[ones(size(Pos1,1),1); zeros(size(HardNegM,1),1) ]); 
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
        
        for j=1 : size(F, 2)
            F(:,j) = step(vision.ContrastAdjuster('OutputRangeSource', 'Property', 'OutputRange', [0,1]), F(:,j));
        end
        
        votes = F * model.w(1:end-1)' + model.w(end);
        Y_hat = reshape(votes, [size(Image{1}, 1), size(Image{1}, 2)]);
        %Y_hat = classRF_predict(F,model_class);
        %Y_hat = reshape(Y_hat, size(Image{1}));
        %Y_hat = 
        data.score{i} = Y_hat;
    end
    % results are accumulated in the cross-validation process
end

%%Output visulisations
for i = 1:length(data.image_filename)
    Y_hat = data.score{i};
    scale_Y_hat = step(vision.ContrastAdjuster, data.score{i}); %Contrast scale the certainties image
    imwrite(scale_Y_hat, fullfile(Y_hat_dir, [data.name{i} '.jpg'])); %Store the certainty image
end

%% evaluate 
[data.dist, data.performance_score, data.misclassification_rate, data.f1] = evaluate_quantative( data );
%Store performance data
evaluation_data.dist = data.dist; evaluation_data.performance_score=data.performance_score;
save([Y_hat_dir, '\evaluation_data'], 'evaluation_data');
save([Y_hat_dir, '\data'], 'data');
save([Y_hat_dir, '\param'], 'param');
plot_evaluation(data);

%% remove feature files
delete([feature_dir, '\*']);

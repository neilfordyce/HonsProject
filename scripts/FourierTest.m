function [ mdl ] = FourierTest( POS_SOURCE_PATH, NEG_SOURCE_PATH )
%FOURIERTEST Summary of this function goes here
%   Detailed explanation goes here
    pos_training = buildTrainingSet(POS_SOURCE_PATH);
    neg_training = buildTrainingSet(NEG_SOURCE_PATH);
    trainingSet = [pos_training; neg_training];
    
    Y = [ones(size(pos_training, 1), 1); ones(size(neg_training, 1), 1) * -1];
    
    mdl = ClassificationKNN.fit(trainingSet, Y, 'NumNeighbors', 5);
end

function [O] = buildTrainingSet(SOURCE_PATH)
    filenames = dir(fullfile(SOURCE_PATH, '*.jpg'));    

    for j = 1 : size(filenames, 1),
        I = imread(fullfile(SOURCE_PATH, filenames(j).name));
        I = rgb2gray(I);
        
        absI = abs(fft2(I));
        absI = absI(:);
        absI = sort(absI, 'descend');
        absI = absI(1:200); %%Select top 2000 frequencies
        
        O(j, :) = absI;
    end
end
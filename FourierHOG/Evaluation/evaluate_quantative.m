%%AUTHOR Neil Fordyce
%%DATE   07/02/14
function [ dist, area_score, misclassification_rate, f1 ] = evaluate_quantative( data )
%EVALUATE_QUANTATIVE produces normalised histograms of frequncy of SVM
%score at pos, neg and ambiguous in three separate histograms.
%Also computes the area trapped between pos and neg histograms to give a
%measure of performance.
BINS = 256;
pos_golgi = [];
ambiguous = [];
neg_golgi = [];

%For each probability image, select the values at the pixels corresponding
%to the trimap groundtruth masks.  Collect them into 3 different lists.
for i=1:numel(data.score)
    pos_golgi = [pos_golgi; data.score{i}(data.mask{i} == 2)];
    ambiguous = [ambiguous; data.score{i}(data.mask{i} == 1)];
    neg_golgi = [neg_golgi; data.score{i}(data.mask{i} == 0)];
end

%Histogram the values
[dist.pos_hist, dist.pos_cen] = norm_hist(pos_golgi, BINS);
[dist.amb_hist, dist.amb_cen] = norm_hist(ambiguous, BINS);
[dist.neg_hist, dist.neg_cen] = norm_hist(neg_golgi, BINS);

[area_score,optimal_thresh] = area(dist.pos_hist, dist.pos_cen, dist.neg_hist, dist.neg_cen);

FN = 0;
FP = 0;
TP = 0;
TN = 0;
total_classified = 0;

for i=1:numel(data.score)
    TP = TP + sum(sum(data.score{i} >= optimal_thresh & data.mask{i} == 2)); %pos above thresh
    FN = FN + sum(sum(data.score{i} < optimal_thresh & data.mask{i} == 2)); %pos below thresh
    FP = FP + sum(sum(data.score{i} >= optimal_thresh & data.mask{i} == 0)); %neg above thresh 
    TN = TN + sum(sum(data.score{i} < optimal_thresh & data.mask{i} == 0)); %neg above thresh 
    total_classified = total_classified + numel(data.score{i});
end

B=1;
f1 = ((1+B^2)*TP)/(((1+B^2)*TP)+((B^2)*FN)+FP);
misclassification_rate = (FN+FP)/total_classified;

end

function [outhist, cen] = norm_hist(image, bins)
%Make a normalised histogram
    [outhist, cen] = hist(image, bins);
    outhist = outhist ./ sum(outhist);
end

function [area_score, optimal_thresh] = area(pos_hist, pos_cen, neg_hist, neg_cen)
%Finds the area trapped between pos and neg histograms - minimize to
%optomize the detector performance 

%Find the frequency where histograms intersect
step_size = 0.01;
neg_intercept = 0;
pos_intercept = 0;
%Step along histograms in the range where they both have bins defined
for i=pos_cen(1):step_size:neg_cen(end)
    %Count the number of bins below the threshold
    neg_thresh = numel(neg_cen(neg_cen <= i));
    pos_thresh = numel(pos_cen(pos_cen <= i));
    
    %Find the point where the frequencies are closest
    neg_freq = neg_hist(neg_thresh);
    pos_freq = pos_hist(pos_thresh);
    freq_diff = neg_freq - pos_freq;
    
    %Set up var to store the value of the last freq_diff on the first iteration
    if ~ exist('last_freq_diff', 'var')
        last_freq_diff = freq_diff;
    end
    
    %Intersection occurs if the difference between the two freqs changes
    %sign from the last iteration, or if they are equal
    if freq_diff * last_freq_diff <= 0
        optimal_thresh = i;
        pos_intercept = pos_thresh;
        neg_intercept = neg_thresh;
    end
    
    %Store for the next iteration
    last_freq_diff = freq_diff;
end

%Crop the histograms to enclose only the area trapped between them
pos_hist = pos_hist(1:pos_intercept);
neg_hist = neg_hist(neg_intercept:end);

%Display the cropped histograms
%{
pos_cen = pos_cen(1:pos_intercept);
neg_cen = neg_cen(neg_intercept:end);
pos_intercept
neg_intercept
figure
plot(pos_cen, pos_hist); hold on; plot(neg_cen, neg_hist);
%}

%Sum to get the area
%1 is worst case performance - fully overlapped, 0 is ideal
area_score = sum([pos_hist, neg_hist]);

end

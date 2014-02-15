%%AUTHOR Neil Fordyce
%%DATE   07/02/14
function [ dist, area_score ] = evaluate_quantative( data )
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

area_score = area(dist.pos_hist, dist.pos_cen, dist.neg_hist, dist.neg_cen);

end

function [outhist, cen] = norm_hist(image, bins)
%Make a normalised histogram
    [outhist, cen] = hist(image, bins);
    outhist = outhist ./ sum(outhist);
end

function [area_score] = area(pos_hist, pos_cen, neg_hist, neg_cen)
%Finds the area trapped between pos and neg histograms - minimize to
%optomize the detector performance 

%Find the frequency where histograms intersect
step_size = 0.01;
freq_diff_min = inf;
neg_intercept = 0;
pos_intercept = 0;
%Step along histograms in the range where they both have bins defined
for i=pos_cen(1):step_size:neg_cen(end)
    %Count the number of bins below the threshold
    neg_thresh = numel(neg_cen(neg_cen <= i)) + 1;
    pos_thresh = numel(pos_cen(pos_cen <= i)) + 1;
    
    %Find the point where the frequencies are closest
    neg_freq = neg_hist(neg_thresh);
    pos_freq = pos_hist(pos_thresh);
    freq_diff = abs(neg_freq - pos_freq);
    
    %If a new min diff is found, update the 
    if freq_diff < freq_diff_min
        freq_diff_min = freq_diff;
        pos_intercept = pos_thresh;
        neg_intercept = neg_thresh;
    end
end

%Crop the histograms to enclose only the area trapped between them
pos_hist = pos_hist(1:pos_intercept);
neg_hist = neg_hist(neg_intercept:end);

%Sum to get the area
area_score = sum([pos_hist, neg_hist]);

%Normalise by number of histograms
%1 is worst case performance - fully overlapped, 0 is ideal
area_score = area_score / 2;

end

function [ dist, pos_golgi, neg_golgi ] = evaluate_quantative( data )
%EVALUATE_QUANTATIVE Summary of this function goes here
%   Detailed explanation goes here
BINS = 256;
pos_golgi = [];
ambiguous = [];
neg_golgi = [];

for i=1:numel(data.score)
    pos_golgi = [pos_golgi; data.score{i}(data.mask{i} == 2)];
    ambiguous = [ambiguous; data.score{i}(data.mask{i} == 1)];
    neg_golgi = [neg_golgi; data.score{i}(data.mask{i} == 0)];
end

[dist.pos_hist, dist.pos_cen] = norm_hist(pos_golgi, BINS);
[dist.amb_hist, dist.amb_cen] = norm_hist(ambiguous, BINS);
[dist.neg_hist, dist.neg_cen] = norm_hist(neg_golgi, BINS);

end

function [outhist, cen] = norm_hist(image, bins)
    [outhist, cen] = hist(image, bins);
    outhist = outhist ./ sum(outhist);
end

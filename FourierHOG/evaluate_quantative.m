function [ dist ] = evaluate_quantative( data )
%EVALUATE_QUANTATIVE Summary of this function goes here
%   Detailed explanation goes here
BINS = 256;
pos_hist = zeros(BINS, 1);
ambiguous_hist = zeros(BINS, 1);
neg_golgi_hist = zeros(BINS, 1);

pos_el = 0;
amb_el = 0;
neg_el = 0;

for i=1:numel(data.score)
    pos_golgi = data.score{i}(data.mask{i} == 2);
    ambiguous = data.score{i}(data.mask{i} == 1);
    neg_golgi = data.score{i}(data.mask{i} == 0);
    
    pos_el = pos_el + sum(pos_golgi);
    amb_el = amb_el + sum(ambiguous);
    neg_el = neg_el + sum(neg_golgi);
    
    pos_hist = pos_hist + imhist(pos_golgi);
    ambiguous_hist = ambiguous_hist + imhist(ambiguous);
    neg_golgi_hist = neg_golgi_hist + imhist(neg_golgi);
end

dist.pos_hist = pos_hist ./ pos_el;
dist.ambiguous_hist = ambiguous_hist ./ amb_el;
dist.neg_hist = neg_golgi_hist ./ neg_el;

%could normalise

end

function hist = normalised_hist(image)
    hist = imhist(image) ./ numel(image);
end

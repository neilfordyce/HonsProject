function [ maxIndex ] = pick_top( w, top )
%PICK_TOP weights so we can retrain using only the most improtant weights.
%Returns a mask showing the location of the top weights in the feature
%vector
sortedWeights = unique(w(:));
maxWeights = sortedWeights(end-top+1:end);
maxIndex = ismember(w, maxWeights);
maxIndex = maxIndex(1:end-1);
end


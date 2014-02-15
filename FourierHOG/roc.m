%%AUTHOR Neil Fordyce
%%DATE   10/02/14
function roc( data )
%ROC Produces RoC curve for data
step_size = -0.01;
neg_cen = data.dist.neg_cen;
pos_cen = data.dist.pos_cen;

%Find the range of scores in the histograms
min_bin = min([neg_cen, pos_cen]);
max_bin = max([neg_cen, pos_cen]);

fp = [];
tp = [];

%Iterate across the entire range of the histogram, decreasing the threshold
%score on each iteration
for i=max_bin:step_size:min_bin
    %Count the number of bins below the threshold
    neg_thresh = numel(neg_cen(neg_cen < i)) + 1;
    pos_thresh = numel(pos_cen(pos_cen < i)) + 1;
    
    %Get all the bins above the threshold
    neg_above_thresh = data.dist.neg_hist(neg_thresh:end);  %the last neg_thresh bins
    pos_above_thresh = data.dist.pos_hist(pos_thresh:end);  %the last pos_thresh bins
    
    %Get the total tp and fp at this thresh, by summation
    fp = [fp, sum(neg_above_thresh)];
    tp = [tp, sum(pos_above_thresh)];
end

%Plot the RoC curve
figure
plot(fp, tp)
axis tight
xlabel('FP Rate')
ylabel('TP Rate')
    
end

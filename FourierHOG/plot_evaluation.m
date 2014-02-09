function [] = plot_evaluation( data )
%PLOT_EVALUATION Summary of this function goes here
%   Detailed explanation goes here
figure
hold on
plot(data.dist.neg_cen, data.dist.neg_hist, 'r')
plot(data.dist.amb_cen, data.dist.amb_hist, 'b')
plot(data.dist.pos_cen, data.dist.pos_hist, 'g')
legend('neg','ambiguous','pos')
end

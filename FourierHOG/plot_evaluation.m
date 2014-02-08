function [ output_args ] = plot_evaluation( data )
%PLOT_EVALUATION Summary of this function goes here
%   Detailed explanation goes here
figure
hold on
plot(data.dist.neg_hist, 'r')
plot(data.dist.ambiguous_hist, 'b')
plot(data.dist.pos_hist, 'g')
legend('neg','ambiguous','pos')
end


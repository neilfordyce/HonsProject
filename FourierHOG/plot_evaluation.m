%%AUTHOR Neil Fordyce
%%DATE   7/02/14
function [] = plot_evaluation( data )
%PLOT_EVALUATION plots the histograms obtained using evaluate_quantative()
%in one figure
figure
hold on
plot(data.dist.neg_cen, data.dist.neg_hist, 'r')
%plot(data.dist.amb_cen, data.dist.amb_hist, 'b')
plot(data.dist.pos_cen, data.dist.pos_hist, 'g')
%legend('neg','ambiguous','pos')
legend('neg','pos')
xlabel('SVM Score')
ylabel('Normalised Frequency')
end

%AUTHOR Neil Fordyce
function [ output_args ] = plot_seg_eval(data_dir)
%Produces detailed plot of segmentation evaluation data.
%Plots Mean jaccard index, missed segmentations and false segmentations on
%a single bar plot with two y-axes

dirs = dir(data_dir);
mean_ji=[];
false_seg=[];
missed_seg=[];
x_labels = {};
indie_variable = 'Sigma';

for i=1:numel(dirs)-2 %first two are . and .. so skip 'em
    data_path = fullfile(data_dir, dirs(i+2).name, 'performance.mat');
    load(data_path, 'performance');
    mean_ji = [mean_ji, performance.mean_ji];
    false_seg = [false_seg, performance.false_seg];
    missed_seg = [missed_seg, performance.missed_seg];
    x_labels{i} = performance.x_label;
end

set(0,'DefaultAxesFontSize', 12)
space = 5;

%Figure out how many plots along x we need to make
x=1:space:(space*numel(mean_ji))-1;
x2=[x+1;x+2];

% Set these three variables as desired
width = (x(2)-x(1))/5;
orange = [256,100,0]./256;

% Plot
figure
[haxes,hbar1,hbar2]=plotyy(x,mean_ji,x2',[false_seg;missed_seg]', @(x,y) bar(x,y,width/4), @(x,y) bar(x,y,width));

set(haxes(1), 'xtickLabel', x_labels);
set(haxes(2), 'xtickLabel', '' );

%Colour the axes
set(haxes,{'ycolor'},{orange;[67,186,52]./256});
set(hbar1, 'facecolor', orange); 
colormap(haxes(1),[winter(2)])

%Labels and legend
axes(haxes(1)); xlabel(indie_variable); ylabel('Mean Jaccard Index J>0');
axes(haxes(2)); ylabel('Count of False/Missed Segmentations');
%legend({'False Segmentation','Missed Segmentation','Mean Jaccard Index'});%, 'Location', 'Best');

end


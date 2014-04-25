%AUTHOR: Neil Fordyce
function batch_roc( data_dir, figure_name)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
dirs = dir(data_dir);

linespec = {'-r', '-b', '-g', '-k.', '-b*', '-*r', '^g-', '-k^'};

set(0,'DefaultAxesFontSize', 16)
set(0,'defaultlinelinewidth',1)
figure;
hold on
legend_cell = {};

for i=1:numel(dirs)-2 %first two are . and .. so skip 'em
   data_path = fullfile(data_dir, dirs(i+2).name, 'data.mat');
   load(data_path, 'data');
   plot_handle = roc(data, linespec{i});
   
   legend_cell{i} = data.legend_text;
end

%legend_cell{1} = 'with coupling';
%legend_cell{2} = 'without coupling';

legend(legend_cell, 'Location', 'SouthEast');
figure_name = ['C:\Users\Neil\SkyDrive\University\HonoursProject\Reports\', figure_name, '.eps'];
print('-depsc', figure_name);

end


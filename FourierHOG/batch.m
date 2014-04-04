%Make dir to output certainty images
clearvars
close all
dbstop if error

legend_text = 'm=0,1,2'

for order=3:6
    batch_run = 1;
    
    output_dir = 'variable_order';
    load_params
    
    param.maxOrder = order;
    legend_text = [legend_text, ',', num2str(order)];
    
    main    
end

clear('batch_run'); %get rid of the batch flag
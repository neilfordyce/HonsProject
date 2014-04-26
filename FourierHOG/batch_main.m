%%AUTHOR Neil Fordyce
%Basic setup, load params, loop to change a param, call 'main'
%Set legend text for output to roc curve later
%Make a separate output directory for each batch run, and set output_dir to
%it
%Make dir to output certainty images
clearvars
close all
dbstop if error

for scale=[30]
    batch_run = 1;
    
    output_dir = 'variable_scale';
    load_params
    
    param.featureScale = scale;
    legend_text = ['scale=', num2str(scale)];
    
    main    
end

clear('batch_run'); %get rid of the batch flag
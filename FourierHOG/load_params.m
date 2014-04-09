% Training Parameters
param.featureScale = 6;

param.sample_count = 35;	%Number of images to use

param.neg_sample_count = 300;  %Per image

param.pos_sample_multiplier = 5000;

param.scale           = 0.2;         %Need to subsample the images because the 
                                      %electron micrographs are huge
                                      
param.maxOrder = 2;

param.gt_mask_dir = '/output/gt_masks_2';
									  
% Train/Test Data
%TODO this is pretty redundant now, consider removal
%param.cand_threshold   = 0.0; % Only use candidates with
                                   % probability greater than this

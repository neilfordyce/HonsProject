% Training Parameters
param.featureScale = 5; %TODO Increase feature scale with image resize

param.sample_count = 30;

param.neg_sample_count = 3000;  %Per image

param.pos_sample_multiplier = 50000;  %TODO Since the change in GT this shouldn't be so high

param.scale           = 0.2;         %Need to subsample the images because the 
                                      %electron micrographs are huge
									  
% Train/Test Data
%TODO this is pretty redundant now, consider removal
%param.cand_threshold   = 0.0; % Only use candidates with
                                   % probability greater than this

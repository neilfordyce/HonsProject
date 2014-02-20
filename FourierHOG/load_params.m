%TODO quite a few of these are redundant
% Learning Parameters
params.scale           = 0.3;     %Need to scale the images because the 
                                      %electron micrographs are huge

params.class           = 'cars';  % object tag

params.QK              = 10;      % Number of region clusters
                                      % to use

params.EM_RESTARTS     = 1;       % Number of EM restarts to
                                      % use (10 is usually enough)

params.pre_cluster     = 0;       % If this is set to 1,
                                      % precluster the regions
                                      % before learning the
                                      % relationships, otherwise
                                      % learn them jointly

params.structure_score = 'prior'; % Scoring mechanism for the
                                      % structure

params.expected_num_rs = 0;       % E[# of active relationships]
params.stddev_num_rs   = 1;       % STDDEV[# of active r's]

params.diag_cov        = 0;       % 0 = Full Covariance
                                      % 1 = Diagonal Covariance

% Train/Test Data
params.cand_threshold   = 0.0; % Only use candidates with
                                   % probability greater than this

params.truth_threshold  = 0.0; % A candidate is a "true
                                   % positive" if it's overlap with
                                   % a groundtruth is greater than this

params.train_images = 1:2:30;  % The indices of the training images
params.test_images  = 2:2:10;  % The indices of the test images

% Inference Params
infer_params.INIT      = 'rand'; % How to initialize the Gibbs samples

infer_params.SAMPLES   = 5;      % Number of samples (for real
                                 % experiments, you should use 20-50)

infer_params.ITERS     = 5;      % Number of Gibbs iterations per
                                 % samples (for real experiments,
                                 % use 20-100)


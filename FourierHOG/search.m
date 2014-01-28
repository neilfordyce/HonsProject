% Learning Parameters
tas_params.scale           = 0.2;     %Need to scale the images because the 
                                      %electron micrographs are huge

tas_params.class           = 'cars';  % object tag

tas_params.QK              = 10;      % Number of region clusters
                                      % to use

tas_params.EM_RESTARTS     = 1;       % Number of EM restarts to
                                      % use (10 is usually enough)

tas_params.pre_cluster     = 0;       % If this is set to 1,
                                      % precluster the regions
                                      % before learning the
                                      % relationships, otherwise
                                      % learn them jointly

tas_params.structure_score = 'prior'; % Scoring mechanism for the
                                      % structure

tas_params.expected_num_rs = 0;       % E[# of active relationships]
tas_params.stddev_num_rs   = 1;       % STDDEV[# of active r's]

tas_params.diag_cov        = 0;       % 0 = Full Covariance
                                      % 1 = Diagonal Covariance

% Train/Test Data
tas_params.cand_threshold   = 0.0; % Only use candidates with
                                   % probability greater than this

tas_params.truth_threshold  = 0.0; % A candidate is a "true
                                   % positive" if it's overlap with
                                   % a groundtruth is greater than this

tas_params.train_images = 1:2:30;  % The indices of the training images
tas_params.test_images  = 2:2:10;  % The indices of the test images

% Inference Params
infer_params.INIT      = 'rand'; % How to initialize the Gibbs samples

infer_params.SAMPLES   = 5;      % Number of samples (for real
                                 % experiments, you should use 20-50)

infer_params.ITERS     = 5;      % Number of Gibbs iterations per
                                 % samples (for real experiments,
                                 % use 20-100)


function [Feature, Fdetail] = FourierHOG(I, binSize, order)

if ~exist('binSize', 'var') || isempty(binSize),
    binSize = 6;
end

global chKERNEL triKERNEL triKERNEL2
%%
[om,on,~] = size(I);
padSize = (binSize*5);
%% get gradient
[dx,dy] = gradient(I);
complex_g = complex(dx,dy);%conversion to polar-coords of gradient image

complex_g = padarray(complex_g,[padSize,padSize]);
%% project to fourier space
[m,n] = size(complex_g);
%order = [0 1 2 3];% 4];  % only contrast-insensitive? and postive frequncies
f_g = zeros([m,n,numel(order)]);
phase_g = angle(complex_g);
mag_g = abs(complex_g);

% local gradient magnitude normalization, scale = sigma * 2
local_mag_g = sqrt(conv2(mag_g.^2, triKERNEL2, 'same'));
mag_g = mag_g ./ (local_mag_g + 0.001);

for j = 1:numel(order)
    f_g(:,:,j) = exp( -1i * (order(j)) * phase_g) .* sqrt(mag_g);  %root of mag_g is better here
end
f_g(:,:,1) = f_g(:,:,1) * 0.5;

local_mag_g = unPad(local_mag_g, [om,on]);
%% compute regional description by convolutions
center_f_g = zeros(om,on,numel(order));
template = triKERNEL;
c_featureDetail = [];
for j = 1:numel(order)
    center_f_g(:,:,j) = unPad(conv2(f_g(:,:,j), template, 'valid'), [om,on]);
    c_featureDetail = [c_featureDetail; [0,-1,-order(j), 0, -order(j)]];
end

% frequncy control
maxFreq = order(end);
maxFreqSum = order(end);

% count output feature channels
nScale = size(chKERNEL,1);
featureDetail = [];
for s = 1:nScale
    featureDim = 0;
    for freq = -maxFreq:maxFreq
        for j = 1:numel(order)
            ff = -(order(j))+freq;
            if (freq==-2 && order(j)==0)
                now;
            end
            if(ff >= -maxFreqSum && ff <= maxFreqSum && ~(order(j)==0 && freq < 0))
                featureDim = featureDim + 1;
                featureDetail = [featureDetail; [s,-1,-order(j), freq, ff]];
            end
        end
    end
end

% compute convolutions
fHoG = zeros([om,on,featureDim*nScale]);
%fHoG = [];
cnt = 0;
for s = 1:nScale
    for freq = -maxFreq:maxFreq
        template = chKERNEL{s,abs(freq)+1};
        if(freq < 0)
            template = conj(template);
        end
        for j = 1:numel(order)
            ff = -(order(j))+freq;
            if(ff >= -maxFreqSum && ff <= maxFreqSum && ~(order(j)==0 && freq < 0))
                cnt = cnt + 1;
                fHoG(:,:,cnt) = unPad(conv2(f_g(:,:,j), template, 'valid'), [om,on]);
            end
        end
    end
end

%% visualization

% for s = 1:nScale
%     figure('Name',['Features at scale ' , num2str(s)]);
%     set(gcf, 'Color',[1,1,1]);
%     n = ceil(sqrt(featureDim));
%     HH = tight_subplot(n,n,0.02,0.05,0.05);
%
%     for i = 1:featureDim
%         axes(HH(i));imagesc(real(fHoG(:,:,(s-1) * featureDim + i)));axis equal tight off;colormap jet
%     end
% end

%%
fHoG = reshape(fHoG, om*on, size(fHoG,3));

center_f_g = reshape(center_f_g, om*on, size(center_f_g,3));
%% create invariant features
% some features are naturally rotation-invariant
iF_index = featureDetail(:,end) == 0;
iF = fHoG(:, iF_index);
% for complex number
ifreal = false(1, size(iF,2));
for i = 1:size(iF,2)
    ifreal(i) = isreal(iF(:,i));
end
iF = [real(iF), imag(iF(:,~ifreal))];
i_featureDetail = featureDetail(iF_index,:);
i_featureDetail = [i_featureDetail; i_featureDetail(~ifreal,:)];


% generate magnitude features from the non-invariant features
mF = abs([fHoG(:,~iF_index) center_f_g local_mag_g(:)]);
mFdetail = [featureDetail(~iF_index,:) ; c_featureDetail; [0, -1, -1,-1,-1]];


% coupling features across different radius/scales
% here we couple radius 1/2 and radius 2/3
cF = cell(1,2);
cFdetail = cell(2,1);

for i = 1:2
    j = i+1;
    cF{i} = fHoG(:, (i-1) * featureDim + (1:featureDim) ) .* conj( fHoG(:, (j-1) * featureDim + (1:featureDim) )  );
    cF{i} = cF{i} ./ (sqrt(abs(cF{i})) + eps);      % take magnitude but keep the phase
    cFdetail{i} = featureDetail((i-1) * featureDim + (1:featureDim),:);
    cFdetail{i}(:,2) = featureDetail((j-1) * featureDim + (1:featureDim),1);
end

cF = cell2mat(cF);
cFdetail = cell2mat(cFdetail);
% for complex number
ifreal = false(1, size(cF,2));
for i = 1:size(cF,2)
    ifreal(i) = isreal(cF(:,i));
end
cF = [real(cF), imag(cF(:,~ifreal))];
cFdetail =[cFdetail; cFdetail(~ifreal,:)];

%% final output
Feature = [iF mF cF];
Fdetail = [i_featureDetail; mFdetail; cFdetail];

%subsample feature space
%load('maxIndex.mat');
%Feature = Feature(:, maxIndex==1);
%Fdetail = Fdetail(maxIndex==1, :);
end

%featureDetail(:,end) is all of the rotation orders, i.e. m(i) in the paper

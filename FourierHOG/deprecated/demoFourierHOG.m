%% DemoFourierHoG
clearvars
close all

binSize = 6;
padSize = (binSize*6);

I = im2double((imread('traffic_2004.jpg')));
I = I(1:250,51:300,:);
[om,on,z] = size(I);
% requires tight_subplot from http://www.mathworks.com/matlabcentral/fileexchange/27991
prepareKernel(6, 4, 1)
pause; % show cnvolution kernels

%% all intermediate results are exported into images.
%%
figure(1);
imshow(I); axis equal tight off

% requires export_fig from http://www.mathworks.com/matlabcentral/fileexchange/23629-exportfig
addpath('/home/liu/Matlab/export_fig');
mkdir('output');
export_fig('output/I', '-png');

global chKERNEL triKERNEL triKERNEL2
%% get gradient
if(size(I,3) == 3)
    [DX,DY,~] = gradient(I);
    mag = DX .^ 2 + DY .^ 2;
    [~,channel] = max(mag,[],3);
    dx = DX(:,:, 1) .* (channel == 1) + DX(:,:, 2) .* (channel == 2) + DX(:,:, 3) .* (channel == 3);
    dy = DY(:,:, 1) .* (channel == 1) + DY(:,:, 2) .* (channel == 2) + DY(:,:, 3) .* (channel == 3);
    complex_g = complex(dx,dy);
else
    [dx,dy] = gradient(I);
    complex_g = complex(dx,dy);
end

figure(1)
imagesc(dx);
axis equal tight off; colormap jet
export_fig('output/gradient_x', '-png');
figure(1)
imagesc(dy);axis equal tight off
export_fig('output/gradient_y', '-png');

complex_g = padarray(complex_g,[padSize,padSize],0);
%% project to fourier space
[m,n] = size(complex_g);
order = [0 1 2];  % up to 2, only for demo purpose
f_g = zeros([m,n,numel(order)]);
phase_g = angle(complex_g);
mag_g = abs(complex_g);

% local gradient magnitude normalization, scale = sigma * 2
local_mag_g = sqrt(conv2(mag_g.^2, triKERNEL2, 'same'));
mag_g = mag_g ./ (local_mag_g + 0.001);

for j = 1:numel(order)
    f_g(:,:,j) = exp( -1i * (order(j)) * phase_g) .* mag_g;
    figure(1);
    imagesc(unPad(real(f_g(:,:,j)),[om,on]));
    axis equal tight off
    export_fig(['output/fr_', num2str(order(j))], '-png');
    
    figure(1);
    imagesc(unPad(imag(f_g(:,:,j)),[om,on]));
    axis equal tight off
    export_fig(['output/fi_', num2str(order(j))], '-png');
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

% frequncy control, the choice here is only for demo purpose
maxFreq = 2;
maxFreqSum = 2;

% count output feature channels
nScale = size(chKERNEL,1);
featureDetail = [];
for s = 1:nScale
    featureDim = 0;
    for freq = -maxFreq:maxFreq
        for j = 1:numel(order)
            ff = -(order(j))+freq;
            if(ff >= -maxFreqSum && ff <= maxFreqSum && ~(order(j)==0 && freq < 0))
                featureDim = featureDim + 1;
                featureDetail = [featureDetail; [s,-1,-order(j), freq, ff]];
            end
        end
    end
end

% compute convolutions
fHoG = zeros([om,on,featureDim*nScale]);
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

%% visualization of regional features
for i = 1:size(featureDetail,1)
    figure(1);
    imagesc(real(fHoG(:,:,i)));axis equal tight off;colormap jet
    export_fig(['output/rm', num2str(-featureDetail(i,3)),   'k', num2str(featureDetail(i,4)) ,  'j', num2str(featureDetail(i,1)) ], '-png');
    
    imagesc(imag(fHoG(:,:,i)));axis equal tight off;colormap jet
    export_fig(['output/im', num2str(-featureDetail(i,3)),   'k', num2str(featureDetail(i,4)) ,  'j', num2str(featureDetail(i,1)) ], '-png');
end

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

for i = 1:size(i_featureDetail,1)
    figure(1);
    imagesc(real(reshape(iF(:,i), om, on)));axis equal tight off;colormap jet
    export_fig(['output/iF', num2str(i) ], '-png');
end

% generate magnitude features from the non-invariant features
mF = abs([fHoG(:,~iF_index) center_f_g local_mag_g(:)]);
mFdetail = [featureDetail(~iF_index,:) ; c_featureDetail; [0, -1, -1,-1,-1]];

for i = 1:size(mFdetail,1)
    figure(1);
    imagesc(real(reshape(mF(:,i), om, on)));axis equal tight off;colormap jet
    export_fig(['output/mF', num2str(i) ], '-png');
end

% coupling features across different radius/scales
% here we couple radius 1/2 and radius 2/3
cF = cell(1,2);
cFdetail = cell(2,1);
for i = 1
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

for i = 1:size(cFdetail,1)
    figure(1);
    imagesc(real(reshape(cF(:,i), om, on)));axis equal tight off;colormap jet
    export_fig(['output/cF', num2str(i) ], '-png');
end




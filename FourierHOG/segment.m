function [L]=segment(F)
close all

% read an image
im = imread('C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\FourierHOG_Prob73563579217\110511C_1_IPL.jpg');
im = im2double(im);
%load('C:\Users\Neil\golgi_fourier_features\F1.mat', 'F');
F = reshape(F(:, 111), size(im));
F = im2double(F);

em_im = imread('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\110511C_1_IPL.jpg');
em_im = rgb2gray(em_im);
em_im = imresize(em_im, 0.2);
em_im = em_im(100:end-100, 100:end-100);
%F = F(100:end-100, 100:end-100);
im = im(100:end-100, 100:end-100);


sz = size(im);

k = 3; %number of regions

%Apply kmeans to cluster the image into k regions
data = im2double(im);
data = data(:);
[cluster_label c] = kmeans(data, k);

% calculate the data cost per cluster center
Dc = zeros([sz k],'single');
for cluster_i=1:k
    % variance per cluster
    icv=inv(var(data(cluster_label==cluster_i, :)));
    %subtract cluster mean 
    dif = data - c(cluster_i);
    % data cost is minus log likelihood of the pixel to belong to each
    % cluster according to its intensity value
    Dc(:,:,cluster_i) = reshape((dif*icv).*dif./2,sz);
end


%Dc(:,:, 1) = abs(im - mean(im(:)));
%Dc(:,:,2) = 1-Dc(:,:,1);


% smoothness term: 
% constant part
Sc = ones(k) - eye(k);
% spatialy varying part
[Hc Vc] = gradient(im2double(em_im));
%[Hc Vc] = gradient(im2double(em_im), fspecial('gauss',[3 3]), 'symmetric');
%[Hc Vc] = SpatialCues(F);

%cut the graph
%GraphCut('open', DataCost, SmoothnessCost, vC, hC);
%gch = GraphCut('open', Dc, 30*Sc, exp(-Vc*5), exp(-Hc*5));
gch = GraphCut('open', Dc, 70*Sc, exp(-Vc*30), exp(-Hc*30));
%gch = GraphCut('open', Dc, Sc);
[gch L] = GraphCut('expand',gch, 5);
gch = GraphCut('close', gch);

% show results
imshow(em_im);
hold on;
PlotLabels(~L);



%---------------- Aux Functions ----------------%
function ih = PlotLabels(L)

L = single(L);

bL = imdilate( abs( imfilter(L, fspecial('log'), 'symmetric') ) > 0.1, strel('disk', 1));
LL = zeros(size(L),class(L));
LL(bL) = L(bL);
Am = zeros(size(L));
Am(bL) = .5;
ih = imagesc(LL); 
set(ih, 'AlphaData', Am);
colorbar;
%colormap 'jet';

%-----------------------------------------------%
function [hC vC] = SpatialCues(im)
g = fspecial('gauss', [13 13], sqrt(13));
dy = fspecial('sobel');
vf = conv2(g, dy, 'valid');
sz = size(im);

vC = zeros(sz(1:2));
hC = vC;

for b=1:size(im,3)
    vC = max(vC, abs(imfilter(im(:,:,b), vf, 'symmetric')));
    hC = max(hC, abs(imfilter(im(:,:,b), vf', 'symmetric')));
end

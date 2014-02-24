% Author Neil Fordyce
function [L]=segment(F)

em_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\';
prob_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\FourierHOG_Prob73565287217\';
output_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\segment\';
prob_files = dir([prob_dir, '*jpg']);

file_count = length(prob_files);
file_count = 1;

for file_i = 1:file_count
    filename = prob_files(file_i).name;
    im = imread(fullfile(prob_dir, filename));
    em_im = imread(fullfile(em_dir, filename));

    %im = slic_segment('C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\FourierHOG_Prob73565287217\110511C_1_IPL.jpg', 'C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\110511C_1_IPL.jpg');

    % read an image
    %im = imread('C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\FourierHOG_Prob73565287217\110511C_1_IPL.jpg');
    im = im2double(im);
    %load('C:\Users\Neil\golgi_fourier_features\F1.mat', 'F');

    %em_im = imread('C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\110511C_1_IPL.jpg');
    em_im = rgb2gray(em_im);
    em_im = imresize(em_im, 0.2);
    %em_im = im2double(em_im);
    %em_im = em_im(100:end-100, 100:end-100);

    %{
    F = reshape(F, [size(im), 233]);
    %F = F(100:end-100, 100:end-100, :);
    %im = im(100:end-100, 100:end-100);

    F = im2double(F);

    Fr = F(:,:, 111);
    Fi = F(:,:, 112);
    %}

    %{
    This was a stupid idea
        Fc=complex(Fr,Fi);
    angleFc = angle(Fc);
    angleFc = atan(angleFc);
    angleFc = im2double(angleFc);
    %}

    %[Dc, dif]=data_cost_hist(im);
    [Dc, dif]=data_cost(im);
    %[Dc, dif]=data_cost_kmeans(im);

    % smoothness term: 
    % constant part
    Sc = [0 1;
          1 0];
    Sc = single(Sc);
    % spatialy varying part
    blur_em_im = conv2(em_im, fspecial('gauss',[20 20]), 'same');
    [Hc Vc] = gradient(im2double(blur_em_im));
    %Weight the orientations by the certainty
    Hc = (Hc .* dif).^2;  %TODO Maybe subtract mean
    Vc = (Vc .* dif).^2;
    %[Hc Vc] = gradient(Fi);
    %[Hc Vc] = gradient(im2double(em_im), fspecial('gauss',[3 3]), 'symmetric');
    %[Hc Vc] = SpatialCues(im2double(em_im));

    %cut the graph
    %GraphCut('open', DataCost, SmoothnessCost, vC, hC);
    %gch = GraphCut('open', Dc, 30*Sc, exp(-Vc*5), exp(-Hc*5));
    gch = GraphCut('open', Dc, 80*Sc, exp(-Vc*2), exp(-Hc*2)); %data_cost(im)
    %gch = GraphCut('open', Dc, 2*Sc, exp(-Vc*50), exp(-Hc*50)); %data_cost_hist(im)
    %gch = GraphCut('open', Dc, 50*Sc, exp(-Vc*5), exp(-Hc*5)); %data_cost_kmeans(im)
    %gch = GraphCut('open', Dc, 150*Sc);
    [gch L] = GraphCut('expand',gch, 5);
    gch = GraphCut('close', gch);

    % show results
    %imshow(em_im);
    %hold on;
    ih = PlotLabels(L);
    em_im = repmat(em_im, [1, 1, 3]);
    em_im(ih>0)=ih(ih>0);
    imshow(em_im);
    imwrite(em_im, fullfile(output_dir, filename));
end
end

function [Dc, dif] = data_cost(I)
    mean_I = mean(I(:));
    I(I<mean_I) = mean_I;
    dif = (I - mean_I).^2;
    %variance = var(I(:));

    %dif = (dif-min(dif(:))) ./ (max(dif(:)-min(dif(:))));

    Dc(:,:,1) = ((numel(dif)*dif)./(sum(dif(:))*2));
    %Dc(:,:,1) = (dif./(variance*2));
    Dc(:,:,2) = 1-Dc(:,:,1);
end

function [Dc, dif] = data_cost_hist(I)
    dif = I;
    
    mean_I = mean(I(:));
    I(I<mean_I) = mean_I;
    dif = (I - mean_I).^2;
    
    hist_dif = hist(dif(:));
    dif = repmat(I, [1, 1, size(hist_dif, 2)]);

    for i=1:size(dif, 1)
        for j=1:size(dif, 2)
            dif(i,j,:)=dif(i,j) .* hist_dif;
        end
    end

    dif = sum(dif, 3);
    dif = (dif-min(dif(:))) ./ (max(dif(:)-min(dif(:))));

    Dc(:,:,1) = (dif.*2).^2;
    Dc(:,:,2) = 1-Dc(:,:,1); 
end

function [Dc, dif] = data_cost_kmeans(I)
   k = 2; %number of regions

    %Apply kmeans to cluster the image into k regions
    data = im;
    %data = data(:);
    cluster_label = kmeans(data(:), k);

    % calculate the data cost per cluster center
    for cluster_i=1:k
        cluster_data = data(:);
        cluster_data = cluster_data(cluster_label==cluster_i);  %get the data assigned the cluster
        icv = inv(var(cluster_data));                   %intra-cluster variance 

        dif = data - mean(cluster_data);                %subtract cluster mean 

        % data cost is minus log likelihood of the pixel to belong to each
        % cluster according to its intensity value
        Dc(:,:,cluster_i) = (dif*icv).*dif./2;
    end
    %}
    %data = im; 
end

function LL = PlotLabels(L)

L = single(L);

bL = imdilate( abs( imfilter(L, fspecial('log'), 'symmetric') ) > 0.1, strel('disk', 1));
LL = zeros(size(L),class(L));
LL(bL) = L(bL);
Am = zeros(size(L));
Am(bL) = .5;
LL(:,:,2)=LL(:,:,1);
LL(:,:,3)=LL(:,:,1)*0;
LL(:,:,1)=LL(:,:,1)*0;

ih = imagesc(LL); 
set(ih, 'AlphaData', Am);
colorbar;
%colormap 'jet';

end
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
end


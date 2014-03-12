% Author Neil Fordyce
function [performance]=segment()

%TODO parameterise seg file as well
load_params;
performance = {};  performance.false_seg = 0; performance.missed_seg = 0;
accuracy = [];

gt_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\output\gt_masks_2';
em_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\annotated_images\golgi\';
prob_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\FourierHOG_Prob73565931169\';
output_dir = 'C:\Users\Neil\SkyDrive\University\HonoursProject\img\outputs\segment\';
prob_files = dir([prob_dir, '*jpg']);

file_count = length(prob_files);
%file_count = 3;

for file_i = 1:file_count
%for file_i = 14:14
    % read the SVM image, electron micrograph image and ground truth for evaluation
    filename = prob_files(file_i).name;
    im = imread(fullfile(prob_dir, filename));
    em_im = imread(fullfile(em_dir, filename));

    %im = slic_segment(im, em_im);
    im = guard(im);
    im = im2double(im);

    em_im = rgb2gray(em_im);
    em_im = imresize(em_im, 0.2);
    %em_im = im2double(em_im);
    em_im = guard(em_im);
    
    gt = read_mask( fullfile(gt_dir, filename), param.scale);
    gt = guard(gt);

    %[Dc, dif]=data_cost_hist(im);
    [Dc, dif]=data_cost(im);
    %[Dc, dif]=data_cost_kmeans(im);

    % smoothness term: 
    % constant part
    Sc = [0 1;
          1 0];
    Sc = single(Sc);
    % spatialy varying part
    [Hc Vc] = GradientOrientation(im2double(em_im));

%    sparseSc = sparseSmooth(em_im);
    
%    Dc1 = Dc(:, :, 1);
%    Dc2 = Dc(:, :, 2);
%    Dc = [Dc1(:) Dc2(:)]';
    
    %cut the graph
    %GraphCut('open', DataCost, SmoothnessCost, vC, hC);
    %gch = GraphCut('open', Dc, 30*Sc, exp(-Vc*5), exp(-Hc*5));
    %gch = GraphCut('open', Dc*100, 150*Sc, tanh(Vc*0.5), tanh(Hc*0.5)); %data_cost(im)
    gch = GraphCut('open', Dc, 150*Sc, exp(Vc*250), exp(Hc*250)); %data_cost(im)
    %gch = GraphCut('open', Dc, 2*Sc, exp(-Vc*50), exp(-Hc*50)); %data_cost_hist(im)
    %gch = GraphCut('open', Dc, 50*Sc, exp(-Vc*5), exp(-Hc*5)); %data_cost_kmeans(im)
    %gch = GraphCut('open', Dc, 150*Sc);
    [gch L] = GraphCut('expand',gch, 5);
    gch = GraphCut('close', gch);
	
	%Dc1 = Dc(:,:,1);
	%Dc2 = Dc(:,:,2);
	%HcVc = Hc+Vc;
	%L = GCMEX(zeros(size(im(:))), [Dc1(:);Dc2(:)], PAIRWISE, Sc,1);

    L = prune_labels(L);
    
    [acc, F1, missed_seg, false_seg] = evaluate_segment(gt, L);
    accuracy = [accuracy, acc];
    
    performance.false_seg = performance.false_seg + false_seg;
    performance.missed_seg = performance.missed_seg + missed_seg;
        
    %% Overlay labels on image 
    %imshow(em_im);
    %hold on;
    ih = PlotLabels(L);
    em_im = repmat(em_im, [1, 1, 3]);
    em_im(ih>0)=ih(ih>0);
    
    %% Add evaluation scores to bottom
    out_height = 40;
    text_em_im = [em_im; zeros(out_height, size(em_im, 2), size(em_im,3))];
    
    htxtins = vision.TextInserter( sprintf('%#1.4f      ', acc));
    htxtins.Color = [255,255,255]; % [red, green, blue]
    htxtins.FontSize = 24;
    htxtins.Location = [ 1, size(em_im, 1) + out_height/2]; % [x y]
    htxtins.Antialiasing = true;
    htxtins.Font = 'Consolas';
    text_em_im = step(htxtins, text_em_im);
    
    imshow(text_em_im, []);
    imwrite(text_em_im, fullfile(output_dir, filename));
end

performance.accuracy = accuracy;
performance.acc_above_zero = numel(accuracy(accuracy>0));
performance.acc_above_thresh = numel(accuracy(accuracy>0.5));
performance.mean_ji = mean(accuracy(accuracy>0));

end

function [L] = prune_labels(L)
%% Prune labels with area below threshold
    % find the area of all connected comps
    CC = bwconncomp(L);
    STATS = regionprops(CC, 'Area');
    
    Area = [];
    for i=1:numel(STATS)
        Area(i) = STATS(i).Area;
    end
    
    %Threshold the areas
    TotalImageArea = CC.ImageSize(1) * CC.ImageSize(2);
    AreaThresh = TotalImageArea * 0.006;  % Threshold anything below 0.6% of the total image size
    PixelIdxList = CC.PixelIdxList;
    PixelIdxList = PixelIdxList(Area > AreaThresh);
    
    %Remove the regions from the binary label image
    L = zeros(size(L));
    for i=1:numel(PixelIdxList)
       L(PixelIdxList{i})=1; 
    end
end

function [Dc, dif] = data_cost(I)
    %level = graythresh(I);
    mean_I = mean(I(:));
    I(I<mean_I) = mean_I;
    dif = (I - mean_I).^2;
    %variance = var(I(:));

    %dif = (dif-min(dif(:))) ./ (max(dif(:)-min(dif(:))));

    Dc(:,:,1) = ((numel(dif)*dif)./(sum(dif(:))*2));
    %Dc(:,:,1) = 
    %Dc(:,:,1) = (dif./(variance*2));  %TODO Fix the maths below, difficult to see what's going on
    maxDc = Dc(:,:,1);
    maxDc = max(maxDc(:));
    Dc(:,:,2) = 1-Dc(:,:,1)/maxDc;  
    %Dc(:,:,2) = 1-Dc(:,:,1);  
    %Dc(:,:,2) = (Dc(:,:,2).^.75)*2; %higher threshold to reduce false positives
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
set(ih,'AlphaData',Am);
end

function [hC vC] = GradientOrientation(I)
%TODO 
%This can probably be replaced with imgradient function for matlab ver >=2012b
dy = conv2(fspecial('gauss', [5 5], sqrt(13)), fspecial('sobel'), 'valid');
dx = dy';

%abs because direction doesn't matter, just want to cost of finding boundaries 
%where on gradient lines to be less than where there is low gradient
vC = -abs(imfilter(I, dy, 'symmetric'));
hC = -abs(imfilter(I, dx, 'symmetric'));

%vC = -(abs(imfilter(I, dy, 'symmetric')).^2)/(2*var(I(:)));
%hC = -(abs(imfilter(I, dx, 'symmetric')).^2)/(2*var(I(:)));

%vC = min(ones(size(vC))*-0.005,vC);
%hC = min(ones(size(vC))*-0.005,hC);

%vC = 1-(vC/max(vC(:)));
%hC = 1-(hC/max(hC(:)));

%Use of an exponential function allows the gradients to be scaled
%emphasising edges without raising the value of small gradients.  I.e.
%scale in a non-linear fashion

%vC = -((vC - mean(vC(:))).^2)/var(vC(:));
%hC = -((hC - mean(hC(:))).^2)/var(hC(:));

%vC = -step(vision.ContrastAdjuster('OutputRangeSource', 'Property', 'OutputRange', [0,max(abs(vC(:)))]), vC);
%hC = -step(vision.ContrastAdjuster('OutputRangeSource', 'Property', 'OutputRange', [0,max(abs(hC(:)))]), hC);
end

function [SparseSmoothness] = sparseSmooth(I)
    
    I=im2double(I);
    [h,w] = size(I);
    I = I(:);
    variance = var(I);

    i = zeros(size(I));
    j = zeros(size(I));
    s = zeros(size(I));
    
    for k=1:numel(I)
        try
            i(k)=k;
            j(k)=k-1;
            s(k)=(abs(I(k)-I(k-1)).^2)./variance;
        catch
        end
        try
            i(k)=k;
            j(k)=k+1;
            s(k)=(abs(I(k)-I(k+1)).^2)./variance;
        catch
        end
        
        try
            i(k)=k;
            j(k)=k-w;
            s(k)=(abs(I(k)-I(k-w)).^2)./variance;
        catch
        end
        try
            i(k)=k;
            j(k)=k+w;
            s(k)=(abs(I(k)-I(k+1)).^2)./variance;
        catch
        end

    end
    k
    SparseSmoothness = sparse(i,j,exp(-s*250));  
    SparseSmoothness=SparseSmoothness(:,1:numel(I));
    %SparseSmoothness = sparse(SparseSmoothness);
    %{
    #####
    #####
    #####
    #####
    #####
    
    ##### ##### ##### ##### #####
    %}
    
end


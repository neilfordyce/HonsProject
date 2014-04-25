%% Author: Neil Fordyce
% Segmentation using graph cuts
function [em_im, L] = segment(svm_im, em_im, seg_param)
if not(exist('seg_param','var'))
    load_seg_params     %seg params
end

    %% Calculate data cost
%[cost, dif]=data_cost_hist(im);
    [cost, dif]=data_cost(svm_im);

    %% smoothness costs
    % static part
    smoothing_cost = [0 1;
                      1 0];
    smoothing_cost = single(smoothing_cost);
    % spatialy varying part
    [h_cost v_cost] = gradient_orientation(im2double(em_im), seg_param.gradient_smoothing_sigma, seg_param.diff_kernel);
    
    %% Approx energy minimisation
    gch = GraphCut('open', cost, seg_param.lambda*smoothing_cost, exp(v_cost*250), exp(h_cost*250)); %data_cost(im)
    [gch L] = GraphCut('expand',gch, 5);
    gch = GraphCut('close', gch);

    %% Post processing
    L = prune_labels(L);
    
    L=guard(L);
    em_im=guard(em_im);
        
    %% Overlay labels on image 
    %imshow(em_im);
    %hold on;
    ih = show_labels(L);
    em_im = repmat(em_im, [1, 1, 3]);
    em_im(ih>0)=ih(ih>0);
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
    AreaThresh = TotalImageArea * 0.01;  % Threshold anything below 0.6% of the total image size
    PixelIdxList = CC.PixelIdxList;
    PixelIdxList = PixelIdxList(Area > AreaThresh);
    
    %Remove the regions from the binary label image
    L = zeros(size(L));
    for i=1:numel(PixelIdxList)
       L(PixelIdxList{i})=1; 
    end
end

function [cost, dif] = data_cost(I)
    %level = graythresh(I);
    %TODO remove thresh?
    mean_I = mean(I(:));
    I(I<mean_I) = mean_I;
    dif = (I - mean_I).^2;
    %variance = var(I(:));

    %dif = (dif-min(dif(:))) ./ (max(dif(:)-min(dif(:))));

    cost(:,:,1) = ((numel(dif)*dif)./(sum(dif(:))*2));
    %cost(:,:,1) = 
    %cost(:,:,1) = (dif./(variance*2));  %TODO Fix the maths below, difficult to see what's going on
    maxcost = cost(:,:,1);
    maxcost = max(maxcost(:));
    cost(:,:,2) = 1-cost(:,:,1)/maxcost;  %Opposite cost to go to other labelling
    %cost(:,:,2) = 1-cost(:,:,1);  
    %cost(:,:,2) = (cost(:,:,2).^.75)*2; %higher threshold to reduce false positives
end

function [cost, dif] = data_cost_hist(I)
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

    cost(:,:,1) = (dif.*2).^2;
    cost(:,:,2) = 1-cost(:,:,1); 
end

function LL = show_labels(L)

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

function [hC vC] = gradient_orientation(I, sigma, diff_kernel)
%TODO 
%This can probably be replaced with imgradient function for matlab ver >=2012b
dy = conv2(fspecial('gauss', [5 5], sigma), diff_kernel, 'valid');  %combine the sobel and gauss kernels
dx = dy';

%abs because direction doesn't matter, just want the cost of crossing boundaries 
%where on gradient lines to be less than where there is low gradient
vC = -abs(imfilter(I, dy, 'symmetric'));
hC = -abs(imfilter(I, dx, 'symmetric'));

%vC = vC/(2*var(I(:)));
%hC = hC/(2*var(I(:)));

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
end


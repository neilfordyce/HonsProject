mask = imread('110511C_2_IPL_Golgi_region.jpg');
mask = rgb2gray(mask);
mask = 255 - mask;
mask(mask > 1) = 1;
mask(mask < 1) = 0;
imshow(mask);
function output_mask = slic_segment(svm_I, em_I)

svm_I = imread(svm_I);
em_I = imread(em_I);
em_I = rgb2gray(em_I);
em_I = imresize(em_I, 0.2);
svm_I = im2double(svm_I);
em_I = im2double(em_I);

segments = vl_slic(single(em_I), 20, 100);

output_mask = zeros(size(em_I));

for seg=0:max(segments(:))
   segment = svm_I( segments==seg );
   %if sum(segment)/numel(segment) > 0.6
       output_mask( segments==seg ) = mean(segment);
   %end
end

output_mask = reshape(output_mask, size(em_I));

imshow(output_mask)

end
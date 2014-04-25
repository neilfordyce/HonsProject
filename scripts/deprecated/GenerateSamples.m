function [ O ] = GenerateSamples( image_path, output_path, PATCH_SIZE, samples )
%Generates random samples from input image

    filenames = dir(fullfile(image_path, '*.png'));

    O = zeros(PATCH_SIZE, PATCH_SIZE, samples, 'uint8');
    total_samples_taken = 0;
    opaque_pixel_counts = [];
    
    %Find the number of opaque pixels in each image, which is used to
    %distribute how many samples to take from each image
    for j = 1 : size(filenames, 1),    
        [I, map, alpha] = imread(fullfile(image_path, filenames(j).name));
        opaque_pixel_counts = [opaque_pixel_counts; numel(alpha(alpha>0))];
    end
    
    for j = 1 : size(filenames, 1),
        filepath = fullfile(image_path, filenames(j).name)
        [I, map, alpha] = imread(filepath);
        [pathstr, filename, ext] = fileparts(filepath);
        I = imread(filepath);
        I = rgb2gray(I);
        
        %Samples required for each image is defined by how many opaque 
        %pixels are in the current image over the total number of opaque
        %pixels in the data set
        samples_required = round((opaque_pixel_counts(j) / sum(opaque_pixel_counts)) * samples)        
        
        %Maximum number of attempts to get a patch before giving up
        attempts = 1000;
        
        while(samples_required > 0 && attempts > 0)
            attempts = attempts - 1;
            
            %Apply random rotation and pick random coordinates for patch
            rot = randi(180);
            rot_alpha = imrotate(alpha, rot);
            rot_alpha_size = size(rot_alpha);
            rand_x = randi(rot_alpha_size(1) - PATCH_SIZE);
            rand_y = randi(rot_alpha_size(2) - PATCH_SIZE);

            %Check the selected patch isn't on a transparent part of the image
            if min(rot_alpha(rand_x:rand_x + PATCH_SIZE - 1, rand_y:rand_y + PATCH_SIZE - 1)) == 255;
                %Output the found patch
                total_samples_taken = total_samples_taken + 1;
                rot_I = imrotate(I, rot);
                patch = rot_I(rand_x:rand_x + PATCH_SIZE - 1, rand_y:rand_y + PATCH_SIZE - 1);
                OutputRotate(output_path, filename, patch, samples_required);
                O(:, :, total_samples_taken) = patch;
                samples_required = samples_required - 1;
                attempts = 1000;
            end
        end
    end
end


function OutputRotate(output_path, filename, I, sample_index )
%Store the image I 4 times with different rotation each time
    rot = [0, 90, 180, 270];
    
    for j = 1 : size(rot, 2),
        I = imrotate(I, rot(j));
        imwrite(I, fullfile(output_path, [filename, '_', int2str(sample_index), '_', int2str(j),'.jpg']), 'Quality', 100);
    end
end

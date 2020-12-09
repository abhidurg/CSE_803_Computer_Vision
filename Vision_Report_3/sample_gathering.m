
positive_sample_cell = {};
fid = fopen('bbox.txt', 'r');
negative_sample_cell = {};


image_count = 0;
sample_cell = {};

tline = fgetl(fid);
while ~feof(fid)
    if tline(1) == 't' %new image
        image_count = image_count + 1;
        sample_cell{image_count, 1} = image_count;
        sample_cell{image_count, 2} = tline(7:end); %location
         tline = fgetl(fid); %this line is valid and exists, number of faces
         sample_cell{image_count, 3} = tline; %number of faces
         tline = fgetl(fid); %since there is always a face, this line is always valid and exists, bounding box coordinates
         while tline(1) ~= 't' 
             p_box = str2num(tline);
             p_box = round(p_box);
            
             positive_sample_cell{end+1} = p_box;
             %neg_x = p_box(1) + 120;
             %neg_y = p_box(2) + 120;
             %n_box = [neg_x, neg_y, p_box(3), p_box(4)];
             %overlapRatio = bboxOverlapRatio(p_box, n_box);
             %negative_sample_cell{end+1} = overlapRatio;
             if feof(fid)
                break;
             end
             tline = fgetl(fid); 
         end
         sample_cell{image_count, 4} = positive_sample_cell;
         positive_sample_cell = {};
    end
end

%{
for i=1: size(sample_cell,1) %%%LOOP FOR POSITIVE IMAGES 24x24
    rw = imread(sample_cell{i,2}); %read each image
    %gw = rgb2gray(rw);
    if size(rw,3) == 3 %check if image is rgb
        rw = rgb2gray(rw);
    end
    
    bbox_set = sample_cell{i,4};
    for j=1: size(bbox_set, 2) %each face/bounding box for each image
        %small_gw = gw(bbox_set{j}(2):bbox_set{j}(2)+bbox_set{j}(4),bbox_set{j}(1):bbox_set{j}(1)+ bbox_set{j}(3));
        bound_rw = imcrop(rw, bbox_set{j});
        resized_bound_rw = imresize(bound_rw, [24,24]);
        new_directory = strcat(sample_cell{i,2}(1:end-4), '_', int2str(j), '.jpg');
        %imwrite(resized_bound_rw, new_directory); %%UNCOMMENT THIS LINE TO
        %ACTUALLY WRITE THE IMAGES TO YOUR DIRECTORY
    end
end
%}




for i=1: size(sample_cell,1)
    do_not_copy = 0;
    rw = imread(sample_cell{i,2}); %read each image
    %gw = rgb2gray(rw);
    if size(rw,3) == 3 %check if image is rgb
        rw = rgb2gray(rw);
    end
    
    bbox_set = sample_cell{i,4};
    wid = bbox_set{1}(3);
    hei = bbox_set{1}(4);
    
    negative_bound_rw = imcrop(rw, [75, 0, wid, hei]);
    
    for j=1: size(bbox_set, 2) %each face/bounding box for each image
        %small_gw = gw(bbox_set{j}(2):bbox_set{j}(2)+bbox_set{j}(4),bbox_set{j}(1):bbox_set{j}(1)+ bbox_set{j}(3));

        %r_max = size(rw,1);
        %r_rand = randi([1,r_max]);
        %c_max = size(rw, 2);
        %c_rand = randi([1,c_max]);
        
        %negative_bound_rw = imcrop(rw, [r_rand, c_rand, wid, hei]);
        
        %bound_rw = imcrop(rw, bbox_set{j});
        overlapRatio = bboxOverlapRatio([75, 0, wid, hei], bbox_set{j});
        if overlapRatio > 0
            do_not_copy = 1;
        end
        resized_bound_rw = imresize(bound_rw, [24,24]);
        
        %imwrite(resized_bound_rw, sample_cell{i,2});
    end
    
    if do_not_copy == 0
        resized_neg_bound_rw = imresize(negative_bound_rw, [24,24]);
        new_directory = sample_cell{i,2};
        %new_directory = strcat('C:\Users\abhir\Documents\MATLAB\cse803_Hw3_prob3_dataset\dataset\train\negat\', sample_cell{i,2});
        imwrite(resized_neg_bound_rw, new_directory);
    end
end


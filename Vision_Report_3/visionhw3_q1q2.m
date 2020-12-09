rgb_image = imread("14.jpg");
red_chan = rgb_image(:,:,1);
green_chan = rgb_image(:,:,2);
blue_chan = rgb_image(:,:,3);

row_length = size(red_chan, 1);
column_length = size(red_chan, 2);

red_chan_binary = cell(row_length, column_length);
green_chan_binary = cell(row_length, column_length);
blue_chan_binary = cell(row_length, column_length);
histo_cell_bin = cell(row_length, column_length);
histo_cell_dec = cell(row_length, column_length);


for i=1:row_length
    for j=1:column_length
        red_chan_binary{i,j} =  dec2bin(red_chan(i,j), 8);
        green_chan_binary{i,j} =  dec2bin(green_chan(i,j), 8);
        blue_chan_binary{i,j} =  dec2bin(blue_chan(i,j), 8);
    end
end

for i=1:row_length
    for j=1:column_length
        red_bin = dec2bin(red_chan(i,j), 8);
        red_concat = red_bin(1:2);
        green_bin = dec2bin(green_chan(i,j), 8);
        green_concat = green_bin(1:2); 
        blue_bin = dec2bin(blue_chan(i,j), 8);
        blue_concat = blue_bin(1:2);
        histo_cell_bin{i,j} = strcat(red_concat, green_concat, blue_concat);
        histo_cell_dec{i,j} = bin2dec(histo_cell_bin{i,j});
    end
end

%h = histogram(cell2mat(histo_cell_dec)); %PROBLEM 1


range = 10;
training_cell = {[236,190,167], [219, 173, 158], [143, 87, 62], [239, 206, 171], [137, 75, 50], [178,131,105], [233, 209, 197], [213, 145, 124], ...
    [250, 198, 165], [181, 130, 99], [160, 114, 86], [111, 55, 40], [130,71,46], [216,  172, 145], [194,150,123], [125, 104, 77], [146, 127, 97]...
    [181,166,145], [178, 168, 156], [181,117,115], [243,218,214], [183,126,117], [247, 187, 187], [255, 212,212], [222, 178, 107]};
size_training = size(training_cell);
bw = zeros(row_length, column_length);


for i=1:row_length
    for j=1:column_length
        for k=1:size_training(1,2)
            current_sample = training_cell{1, k};
            current_red = current_sample(1,1);
            current_green = current_sample(1,2);
            current_blue = current_sample(1,3);  
            if (red_chan(i,j) <= current_red+range  && red_chan(i,j) >= current_red-range) && (green_chan(i,j) <= current_green+range  && green_chan(i,j) >= current_green-range) && (blue_chan(i,j) <= current_blue+range && blue_chan(i,j) >= current_blue-range)
                bw(i,j) = 1;
            end
        end
    end
end

%imshow(bw*255);  %PROBLEM 2 of HW 3 - show





labeled_matrix = bw;
row_length = size(labeled_matrix, 1);
column_length = size(labeled_matrix, 2);
current_label = 0;
max_label = 0;
is_labeled = zeros(row_length, column_length);
equivalency_matrix = zeros(2, 1);


for i=1:row_length
    for j=1:column_length
        if bw(i,j) == 1 && i > 1 && j > 1 && j < column_length %we only label non 0 values and if indices are not out of bounds
           if labeled_matrix(i, j-1) == 0 && labeled_matrix(i-1,j) == 0 && labeled_matrix(i-1,j-1) == 0 && labeled_matrix(i-1, j+1) == 0 %if all 4 neighbors are 0 and are therefore nonlabeled, this is new
               max_label = max_label + 1;
               equivalency_matrix(1, max_label)= max_label;
               equivalency_matrix(2, max_label)= max_label;
               labeled_matrix(i,j) = max_label;
               is_labeled(i,j) = 1;
           else %neighbors are not all 0 
                if is_labeled(i, j-1) == 1 && is_labeled(i-1,j) == 1 && is_labeled(i-1,j-1) == 1  && is_labeled(i-1,j+1) == 1 % all 4 neighbors are labeled
                    if labeled_matrix(i, j-1) == labeled_matrix(i-1,j) == labeled_matrix(i-1,j-1) == labeled_matrix(i-1,j+1) %all 4 labels are the same
                        current_label = labeled_matrix(i, j-1);
                        labeled_matrix(i,j) = current_label;
                        is_labeled(i,j) = 1;
                    else %neighbors are labeled, but they are NOT the same, i.e there is a conflict
                        current_label = min([labeled_matrix(i, j-1),labeled_matrix(i-1,j), labeled_matrix(i-1,j-1), labeled_matrix(i-1,j+1)]); % take the minimum of 4 labels
                        
                        for k=1:4 %look at all 4 neighbors
                            if minimum_array(k) ~= 0 && minimum_array(k) ~= current_label %if neighbor is not labeled or it isn't the minimum, need to update equivalency
                                equivalency_matrix(2,minimum_array(k)) = current_label;
                            else
                            end
                    
                        end
                        labeled_matrix(i,j) = current_label;
                        is_labeled(i,j) = 1;
                    end
                    
                else %all neighbors are NOT labeled, so only one or some are labeled here. may or may not be conflict
                    minimum_array = [labeled_matrix(i, j-1),labeled_matrix(i-1,j), labeled_matrix(i-1,j-1), labeled_matrix(i-1,j+1)];
                    current_label = min(minimum_array(minimum_array > 0)); % take the minimum of 4 labels that is greater than 0 since 0 means non-labeled
                    for k=1:4 %look at all 4 neighbors
                            if minimum_array(k) ~= 0 && minimum_array(k) ~= current_label %if neighbor is not labeled or it isn't the minimum, need to update equivalency
                                equivalency_matrix(2,minimum_array(k)) = current_label;
                            else
                            end
                    
                    end
                    labeled_matrix(i,j) = current_label;
                    is_labeled(i,j) = 1;
                end
                
           end
            
        else
            
        end
    end
end

%replace according to equivalency;

for a=1:size(equivalency_matrix, 2)
    if equivalency_matrix(1, a) ~= equivalency_matrix(2, a)
        labeled_matrix(labeled_matrix==equivalency_matrix(1, a))= equivalency_matrix(2, a);
    else
    end
end

blob_perimeter = 0;
interior_pixels = 0;
blob_area = 0;
row_sum = 0;
column_sum = 0;
row_center = 0;
column_center = 0;
output_matrix = zeros(10, max_label);


for b=1:max_label %iterate thru all labels
    for c=1:row_length
        for d=1:column_length
            if labeled_matrix(c,d) == b %label matches the one we want to get the area, centroid, and moments for.
                blob_area = blob_area + 1;
                row_sum = row_sum + c;
                column_sum = column_sum + d;
                %if labeled_matrix(c, d-1) == b && labeled_matrix(c-1,d) == b && labeled_matrix(c-1,d-1) == b ...
                    % && labeled_matrix(c-1, d+1) == b && labeled_matrix(c, d+1) == b &&  labeled_matrix(c+1, d-1) == b ...
                     %&&  labeled_matrix(c+1, d) == b &&  labeled_matrix(c+1, d+1) == b %all surrounding pixels are same label, i.e this is a NOT perimeter pixel
                      %blob_perimeter = blob_perimeter + 0; %DO NOTHING
                      %interior_pixels = interior_pixels + 1;
               % else
                   % if labeled_matrix(c, d-1) == b || labeled_matrix(c-1,d) == b ||  labeled_matrix(c, d+1) == b  ||  labeled_matrix(c+1, d) == b %verticall/horizontally adjacent
                    %    blob_perimeter = blob_perimeter + 1;
                    %end
                    %if labeled_matrix(c-1, d-1) == b || labeled_matrix(c-1,d+1) == b ||  labeled_matrix(c+1, d-1) == b  ||  labeled_matrix(c+1, d+1) == b %diagnoal pixels
                    %    blob_perimeter = blob_perimeter + sqrt(2);
                    %end
  
                %end
                
            else
            end
        end
    end
    row_center = row_sum/blob_area;
    column_center = column_sum/blob_area;
    
    output_matrix(1,b) = b;
    output_matrix(2,b) = blob_area;
    output_matrix(3,b) = row_center;
    output_matrix(4,b) = column_center;
    output_matrix(10,b) = blob_perimeter;
    blob_area = 0;
    row_sum = 0;
    column_sum = 0;
    blob_perimeter = 0;
    interior_pixels = 0;
end

second_order_row_sum = 0;
second_order_row_moment = 0;
second_order_column_sum = 0;
second_order_column_moment = 0;
second_order_mixed_sum = 0;
second_order_mixed_moment = 0;
major_axis = 0;
minor_axis = 0;

for e=1:max_label
    for f=1:row_length
        for g=1:column_length
            if labeled_matrix(f,g) == e
                second_order_row_sum = second_order_row_sum + (f - output_matrix(3,e))^2;
                second_order_column_sum = second_order_column_sum + (g - output_matrix(4,e))^2;
                second_order_mixed_sum = second_order_mixed_sum + ((f - output_matrix(3,e))*(g - output_matrix(4,e)));
            else
            end
            
        end
    
    end
    second_order_row_moment = second_order_row_sum/(output_matrix(2,e));
    second_order_column_moment = second_order_column_sum/(output_matrix(2,e));
    second_order_mixed_moment = second_order_mixed_sum/(output_matrix(2,e));
    
    output_matrix(5,e) = second_order_row_moment;
    output_matrix(6,e) = second_order_column_moment;
    output_matrix(7,e) = second_order_mixed_moment;
   
    if second_order_mixed_moment  == 0 && second_order_row_moment > second_order_column_moment
        major_axis = -90;
        minor_axis = 0;

    else
        if second_order_mixed_moment  == 0 && second_order_row_moment <= second_order_column_moment
            major_axis = 0;
            minor_axis = -90;
        else %the formula can be used now that the two cases are not satisfied. rr =/= cc, therefore we cannot divide by 0
            if second_order_mixed_moment  == 0 && second_order_row_moment <= second_order_column_moment
                major_axis = 180*(atan(-2*second_order_mixed_moment*second_order_row_moment - second_order_column_moment + sqrt((second_order_row_moment - second_order_column_moment)^2 + 4*second_order_mixed_moment^2)))/pi;
                minor_axis = major_axis - 90;
            else
                major_axis = 180*(atan((2*second_order_mixed_moment)/(second_order_row_moment - second_order_column_moment)))/(2*pi);
                %major_axis = (180/pi)*(atan(sqrt(second_order_column_moment + second_order_row_moment + sqrt((second_order_column_moment - second_order_row_moment)^2 + 4*second_order_mixed_moment^2))/(-2*second_order_mixed_moment)));
                minor_axis = major_axis - 90;   
            end
        end     
    end
    
    output_matrix(8,e) = major_axis;
    output_matrix(9,e) = minor_axis;
    

    second_order_row_sum = 0;
    second_order_row_moment = 0;
    second_order_column_sum = 0;
    second_order_column_moment = 0;
    second_order_mixed_sum = 0;
    second_order_mixed_moment = 0;
    major_axis = 0;
    minor_axis = 0;
end
output_matrix = output_matrix(:,all(~isnan(output_matrix))); % removes all of the labels used that ended up not actually being separate components


%simplest, but inaccurate way is saying face is largest area of connected
%component

max_area = max(output_matrix(2,:));
for i=1: size(output_matrix,2)
    if output_matrix(2,i) == max_area
        face_label = output_matrix(1,i);
    end
end

[face_row, face_column] = find(labeled_matrix==face_label);

bounding_height = max(face_row) - min(face_row);
bounding_width = max(face_column) - min(face_column);

if (output_matrix(2,face_label)/(bounding_height * bounding_width) >= 0.4) % high probablity of face if area ratio is over 0.4
    bounded_image = insertShape(rgb_image, 'Rectangle',[min(face_column), min(face_row), bounding_width, bounding_height], 'Color', 'red');
else %low probability of face
    bounded_image = insertShape(rgb_image, 'Rectangle',[min(face_column), min(face_row), bounding_width, bounding_height], 'Color', 'blue');
end

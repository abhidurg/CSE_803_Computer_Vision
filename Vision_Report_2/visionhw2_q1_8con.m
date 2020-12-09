bw = imread("hw2-2B.jpg");
%rgb_bw = imread("shapes.jpg");  UNCOMMMENT THESE 2 LINES AND DELETE THE FIRST LINE TO MAKE MY PROGRAM WORK FOR 'shapes.jpg'. ALSO CHANGE THRESHOLD TO 240
%bw = rgb2gray(rgb_bw);

bw(bw < 60) = 0; %CHANGE THRESHOLD HERE WHEN NEEDED (use 50 for 2A)
bw(bw >= 60) = 1;

 %for i=1:size(bw,1)            %UNCOMMENT THIS BLOCK OF CODE AND CHANGE THRESHOLD TO 140 TO MAKE MY PROGRAM WORK FOR IMAGE 3A
     %for j=1:size(bw,2)
         %if bw(i,j) == 1
           %  bw(i,j) = 0;
         %else
            % bw(i,j) = 1;
         %end
    % end
% end


imshow(bw*255);

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
blob_area = 0;
row_sum = 0;
column_sum = 0;
row_center = 0;
column_center = 0;
output_matrix = zeros(9, max_label);


for b=1:max_label %iterate thru all labels
    for c=1:row_length
        for d=1:column_length
            if labeled_matrix(c,d) == b %label matches the one we want to get the area, centroid, and moments for.
                blob_area = blob_area + 1;
                row_sum = row_sum + c;
                column_sum = column_sum + d;
                %if labeled_matrix(c, d-1) ~= b || labeled_matrix(c-1,d) ~= b || labeled_matrix(c-1,d-1) ~= b ...
                     %|| labeled_matrix(c-1, d+1) ~= b || labeled_matrix(c, d+1) ~= b ||  labeled_matrix(c+1, d-1) ~= b ...
                     %||  labeled_matrix(c+1, d) ~= b ||  labeled_matrix(c+1, d+1) ~= b %atleast one neighboring pixel does not have the same label, i.e this is a perimeter pixel
                    %blob_perimeter = blob_perimeter + 1;
                %else   
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
    
    blob_area = 0;
    row_sum = 0;
    column_sum = 0;
    blob_perimeter = 0;
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

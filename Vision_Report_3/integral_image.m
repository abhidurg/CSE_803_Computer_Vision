function output_image = integral_image(input_image)
%INTEGRAL_IMAGE Summary of this function goes here
%   Detailed explanation goes here
row_length = size(input_image, 1);
column_length = size(input_image, 2);
output_image(1,1) = input_image(1,1);
for i=1: row_length
    for j=1: column_length
        
        if j <= 1
            integral_sum1 = 0;
            if i <= 1
                integral_sum2 = 0;
            else
                integral_sum2 = output_image(i-1,j); 
            end
            integral_sum3 = 0;
        else % j is in range
            if i <= 1
                integral_sum1 = output_image(i,j-1);
                integral_sum2 = 0;
                integral_sum3 = 0;
            else %i is in range, j is in range
                integral_sum1 = output_image(i,j-1);
                integral_sum2 = output_image(i-1,j);
                integral_sum3 = output_image(i-1,j-1);
            end
        end
        
        output_image(i,j) = input_image(i,j) + integral_sum1 + integral_sum2 - integral_sum3;
        
    end
end


end


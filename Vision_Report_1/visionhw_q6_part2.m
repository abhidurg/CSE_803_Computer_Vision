[hw, map] = imread( "hw1.chips1.gif");
hw_final = ind2rgb(hw, map);

new_img = rgb2gray(hw_final);
imwrite(hw_final, "hw1.chips1.tif");
imwrite(hw_final, "hw1.chips1.jpg");

new_img(new_img < 0.4) = 0;
new_img(new_img >= 0.4) = 1;

imshow(new_img);
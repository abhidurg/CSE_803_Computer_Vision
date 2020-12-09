bw = imread("20190911_175620.jpg");
imwrite(bw, "20190911_175620.tif");
gw = rgb2gray(bw);
gw(gw < 60) = 0;
gw(gw >= 60) = 1;
imshow(gw*255);



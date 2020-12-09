positive_training_data = {};
Files=dir('*.*');
for k=3:length(Files)
   FileName = Files(k).name;
   bw = imread(FileName);
   bw = double(bw);
   haar_feature = haar(bw);
   positive_training_data{end+1} = haar_feature;
end
%}


negative_training_data = {};
Files=dir('*.*');
for k=3:length(Files)
   FileName = Files(k).name;
   bw = imread(FileName);
   bw = double(bw);
   haar_feature = haar(bw);
   negative_training_data{end+1} = haar_feature;
end




%load('positive_training_set_all_features.mat');
%load('negative_training_set_all_features.mat');


%positive_or_negative = zeros(1,802);
%positive_or_negative(585:802) = 1;

%{
weight_features = zeros(1, 802);
weight_features(1,1:585) = 1/(2*585);
weight_features(1,586:802) = 1/(2*217);
normalized_weights = zeros(1,802);


for i=1:802 %go thru each image
    normalized_weights(i) = weight_features(i)/(sum(weight_features));
end
%}



%weight_features(1,1:585) = 1/(2*585); %set weight of all positive images to 2m
%weight_features(1,586:) = 1/(2*217); %set weight of all negative images to 2l

%polarity_features = zeros(1, 95348);
%threshold_array = zeros(1,95348);

%MAKING THRESHOLD AND POLARITY ARRAY (ONLY COMPUTE ONCE)
%{
for f=1:95348 %for all the features
    sum_array1 = zeros(1,585);
    %image_count = 0;
    for i=1:585 %for all positive images
        %image_count = image_count + 1;
        sum_array1(1,i) = positive_training_data{i}{1}(f);
    end
    sum_array2 = zeros(1,217);
    for k=1:217 %for all negative images
        %image_count = image_count + 1;
        sum_array2(1,k) = negative_training_data{k}{1}(f);
    end
    pos_mean = mean(sum_array1);
    neg_mean = mean(sum_array2);
    if pos_mean >= neg_mean
        x = neg_mean:0.1:pos_mean;
        polarity_features(1,f) = 1;
    else
        x = pos_mean:0.1:neg_mean;    
        polarity_features(1,f) = -1;
    end
    %y = normpdf(x,pos_mean,std(sum_array1));
    %z = normpdf(x,neg_mean,std(sum_array2));
    %diff = abs(z-y);
    
    %[min_diff, indx] = min(diff);
    %thresh = x(indx); %optimal threshold
    %threshold_array(1,f) = thresh;
end
%}

%CLASSIFICATION

%classifier_array = zeros(802, 95348);
%load('positive_training_set_all_features.mat');
%load('negative_training_set_all_features.mat');
%load('threshold_array_set.mat');
%load('polarity_features_set.mat');
%load('classifier_array_set.mat');
%load('error_rates_set.mat');
%load('weights_set.mat');

%{
for f=1:95348 %for each feature
    for i=1:585 %for each positive image
        if polarity_features(f) == 1 %mean of positive > mean of negative for this feature
            if positive_training_data{i}{1}(f) > threshold_array(f) %correctly classified face as face, make h(x) = 1
                classifier_array(i,f) = 1;
            end
        else %mean of negative > mean of positive for this feature
             if positive_training_data{i}{1}(f) < threshold_array(f) %correctly classified face as face, make h(x) = 1
               classifier_array(i,f) = 1;
            end
        end   
    end
    
    for k=1:217 %for each negativer image
        if polarity_features(f) == 1 %mean of positive > mean of negative for this feature
            %disp(threshold_array(f));
            if negative_training_data{k}{1}(f) < threshold_array(f) %correctly classified non face as non face, make h(x) = 1
               classifier_array(585+k,f) = 1;
            end
        else %mean of negative > mean of positive for this feature
             if negative_training_data{k}{1}(f) > threshold_array(f) %correctly classified face as face, make h(x) = 1
               classifier_array(585+k,f) = 1;
            end
        end   
    end
 
end
%}

%CALCULATE ERROR FOR EACH FEATURE
%{
error_array = zeros(1,95348);
for f=1:95348 %for each features
    error_sum = zeros(1,802);
    for i=1:802 %for each image
        if i < 586 %positive image set
            error_sum(1,i) = normalized_weights(i)*abs(classifier_array(i,f) -  0);
        else
            error_sum(1,i) = normalized_weights(i)*abs(classifier_array(i,f) -  1);
        end 
    end
    error_array(1,f) = sum(error_sum);
end
%}

%CALCULATE MINIMUM ERROR

[minimum_error, indx] = min(error_array(1,:)); %after one rond of boosting, best weak classifier had error/epsilon of 0.2388, an index of 12036
%weakclassifier = zeros(1,10); %Let us start by testing 10 weak classifiers
min_error_array = zeros(1,10);
%indx = 12036;
weakclassifier(1,1) = indx;
min_error_array(1,1) = minimum_error;
classifier_count = 1;
for k=1:10
    %UPDATE WEIGHTS AFTER WEAK CLASSIFICATION
    beta_t = minimum_error/(1-minimum_error);
    for i=1:802 %for each image
        normalized_weights(1+classifier_count, i) = normalized_weights(classifier_count,i)*beta_t^(1-~classifier_array(i,indx));
    end
    %RECLACLULATE ERROR AFTER WEIGHT UPDATE AND FIND BEST CLASSIFIER
    for f=1:95348 %for each features
        error_sum = zeros(1,802);
        for i=1:802 %for each image
            if i < 586 %positive image set
                error_sum(1,i) = normalized_weights(1+classifier_count,i)*abs(classifier_array(i,f) -  0);
            else %negative image set
                error_sum(1,i) = normalized_weights(1+classifier_count,i)*abs(classifier_array(i,f) -  1);
            end 
        end
        error_array(1+classifier_count,f) = sum(error_sum);
    end
    [minimum_error, indx] = min(error_array(1+classifier_count,:));
    while ismember(indx, weakclassifier)
        error_array(1+classifier_count,indx) = 1;
        [minimum_error, indx] = min(error_array(1+classifier_count,:));
    end
    
    min_error_array(1,1+classifier_count) = minimum_error;
    weakclassifier(1,1+classifier_count) = indx;
    classifier_count = classifier_count + 1;
end
% 1 for 10 fold test without pose, 3 for 10 fold test with pose

for mode = [1:2:3]

data = load('FeatureGroupCV.mat');
file_num = 10;

% get features
if mode == 1
    features = cell2mat(data.TrainFeatures); 
else
    temp1 = cell2mat(data.TrainFeatures);
    temp2 = cell2mat(data.TrainPoses);
    temp_sum = sum(temp2, 1);
    temp2 = temp2 ./ repmat(temp_sum, 3, 1);
    features = [temp1; temp2];
end

features = features';
[m1, n1] = size(features);
% get gazes
gazes = cell2mat(data.TrainGazes);
gazes = gazes';

% shuffle total index
total_num = m1;
fold_num = floor(m1 / 10);
total_index = [1 : total_num];
ix = randperm(total_num);
shuffled_total_index = total_index(ix);

overall_mean_degree_difference = 0;
overall_var_degree_difference = 0;
for idx = 1 : file_num
    fidx = num2str(idx-1);
    file_index = strcat('f', fidx);
    
    if mode == 3 % for with pose
        file_index = strcat('with_pose_', file_index);
    else % for without pose
    end
    
    test_index = [1 + (idx-1)*fold_num : idx*fold_num];
    train_index = setdiff([1 : m1], test_index);
    
    % get X1, X2, Y1, Y2
    X1 = features(train_index, :);
    X2 = features(test_index, :);
    Y1 = gazes(train_index, :);
    Y2 = gazes(test_index, :);
    
    neighbor_num = 10;
    [IDX, D] = knnsearch(X1, X2, 'K', neighbor_num);
    % use 1/dis as weight
    D = 1 ./ D;
    temp_sum = sum(D, 2);
    temp_D = repmat(temp_sum, 1, neighbor_num);
    norm_D = D ./ temp_D;
    % compute label by weighted sum
    [m2, n2] = size(Y2);
    label = zeros(m2, 2);
    for i = 1 : m2
        label(i, :) = sum(Y1(IDX(i, :), :) .* repmat(norm_D(i, :)', 1, 2));
    end
    

    % convert shpere coordinates to x-y-z
    y_Y = -sin(Y2(:, 1));
    temp1 = tan(Y2(:, 2));
    x_Y = sqrt((1 - y_Y.^2) ./ (temp1.^2 + 1));
    z_Y = temp1 .* x_Y;

    y_L = -sin(label(:, 1));
    temp2 = tan(label(:, 2));
    x_L = sqrt((1 - y_L.^2) ./ (temp2.^2 + 1));
    z_L = temp2 .* x_L;
    
    idx;
    degree_difference = acos(sum([x_Y, y_Y, z_Y] .* [x_L, y_L, z_L], 2));
    % radian to degree
    degree_difference = degree_difference / pi * 180;
    % theta_difference(isnan(theta_difference)) = [];
    mean_degree_difference = mean(degree_difference);
    var_degree_difference = sqrt(var(degree_difference));
    overall_mean_degree_difference = overall_mean_degree_difference + mean_degree_difference;
    overall_var_degree_difference = overall_var_degree_difference + var_degree_difference;
    
    % write result into files
    out_file = strcat('result_', file_index);
    save(out_file, 'label');
end
% compute overall average theta difference
overall_mean_degree_difference = overall_mean_degree_difference / file_num
overall_var_degree_difference = overall_var_degree_difference / file_num
end
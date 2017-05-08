% 0 for leave one out test without pose, 2 for leave one out test with pose

fnames1 = dir('data/*.mat');

nums = [0, 3000, 6000, 9000, 12000, 15000,...
    18000, 21000, 24000, 27000, 30000,...
    33000, 36000, 39000, 41996, 44996];

real_gaze = [];
predicted_gaze = [];
img_list = [];

for mode = [0: 2 : 2]

data = load('FeatureGroupCV.mat');
file_num = 15;

% get features
if mode == 0
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
result = zeros(2, m1);

overall_mean_degree_difference = 0;
overall_var_degree_difference = 0;
for idx = 1 : file_num
    if idx-1 < 10
        fidx = strcat('0', num2str(idx-1));
    else
        fidx = num2str(idx-1);
    end
    file_index = strcat('p', fidx);
    
    if mode == 2 % for with pose
        file_index = strcat('with_pose_', file_index);
    else % for without pose
    end
    
    if idx < 14
        test_index = [1 + (idx-1)*3000 : idx*3000];
    elseif idx == 14
        test_index = [1 + 13*3000 : 2996 + 13*3000];
    else
        test_index = [1 + 2996 + 13*3000 : 2996 + 14*3000];
    end
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
    z_Y = sqrt((1 - y_Y.^2) ./ (temp1.^2 + 1));
    x_Y = temp1 .* z_Y;

    y_L = -sin(label(:, 1));
    temp2 = tan(label(:, 2));
    z_L = sqrt((1 - y_L.^2) ./ (temp2.^2 + 1));
    x_L = temp2 .* z_L;
    
    idx;
    degree_difference = acos(sum([x_Y, y_Y, z_Y] .* [x_L, y_L, z_L], 2));
    % radian to degree
    degree_difference = degree_difference / pi * 180;

    mean_degree_difference = mean(degree_difference);
    var_degree_difference = sqrt(var(degree_difference));
    overall_mean_degree_difference = overall_mean_degree_difference + mean_degree_difference;
    overall_var_degree_difference = overall_var_degree_difference + var_degree_difference;
    
    result(:, 1+nums(idx) : nums(idx+1)) = label';
    % write result into files
    out_file = strcat('result_', file_index);
    save(out_file, 'label');
end
overall_mean_degree_difference = overall_mean_degree_difference / file_num
overall_var_degree_difference = overall_var_degree_difference / file_num

% for visulization
if mode == 0
    % save leave one out wo pose for further drawing
    save('KNNResult_LeaveOneOut_NHP', 'result');
end
% draw_gaze(real_gaze, predicted_gaze, img_list);
% compute overall average theta difference
end

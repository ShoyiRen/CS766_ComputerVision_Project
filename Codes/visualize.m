% load svr result, lr result
data = load('SVRresult_LeaveOneOut_NHP');
svr_res = data.result;
data = load('KNNResult_LeaveOneOut_NHP');
knn_res = data.result;
data = load('LRresult_LeaveOneOut0');
lr_res = data.result;
groundtruth = data.groundtruth;
features = load('FeatureGroupCV');

nums = [0, 3000, 6000, 9000, 12000, 15000,...
    18000, 21000, 24000, 27000, 30000,...
    33000, 36000, 39000, 41996, 44996];
% choose p00 - p14, each person choose 4 images
img_num = 4;
img_index_list = cell(15, img_num);
svr_res_list = cell(15, img_num);
lr_res_list = cell(15, img_num);
knn_res_list = cell(15, img_num);
groundtruth_list = cell(15, img_num);

for idx = 1 : 15
    % load total data of current person
    if idx-1 < 10
        fidx = strcat('0', num2str(idx-1));
    else
        fidx = num2str(idx-1);
    end
    cur_person_data = load(strcat('New Template/p', fidx, 'features'));
    % current person's feature vectors
    cur_person_features = features.TrainFeatures{idx};
    % first num_imgs left eye features
    part_features = cur_person_features(:, 1 : img_num);
    % get part_svr_res
    part_svr_res = svr_res(:, 1 + nums(idx) : img_num + nums(idx));
    % get lr_res
    part_lr_res = lr_res(:,  1 + nums(idx) : img_num + nums(idx));
    % get knn_res
    part_knn_res = knn_res(:,  1 + nums(idx) : img_num + nums(idx));
    % get gorund truth
    part_groundtruth = groundtruth(:, 1 + nums(idx) : img_num + nums(idx));
    % find correspoing images to be plotted on
    cur_person_total_features = cur_person_data.LeftFeature;
    [m, n]  = size(cur_person_total_features);
    for i = 1 : img_num
        for j = 1 : m
            cur_day_features = cur_person_total_features(j, :);
            cur_day_features = cur_day_features(~cellfun(@isempty, cur_day_features));
            [m2, new_n] = size(cur_day_features);
            for k = 1 : new_n
                if part_features(:, i) == cur_day_features{:, k}
                    img_index_list{idx, i} = [j, k];
                    svr_res_list{idx, i} = part_svr_res(:, i)';
                    lr_res_list{idx, i} = part_lr_res(:, i)';
                    knn_res_list{idx, i} = part_knn_res(:, i)';
                    groundtruth_list{idx, i} = part_groundtruth(:, i)';
                end
            end
        end
    end
end

% read in corresponding images, and store in img_list
img_list = cell(15, img_num);
[m, n] = size(img_index_list);
for i = 1 : m
    for j = 1 : n
        temp = img_index_list{i, j};
        day_index = temp(1);
        index = temp(2);
        % construct file name
        fidx = 'p';
        if i-1 < 10
            fidx = strcat(fidx, '0', num2str(i-1));
        else
            fidx = strcat(fidx, num2str(i-1));
        end
        didx = '/day';
        if day_index < 10
            didx = strcat(didx, '0', num2str(day_index));
        else
            didx = strcat(didx, num2str(day_index));
        end
        fidx = strcat(fidx, didx);
        data = load(fidx);
        images = data.data.left.image;
        img = images(index, :, :);
        img_list{i, j} = reshape(img, 36, 60);
    end
end
% draw gaze
real_gaze = reshape(groundtruth_list, 15*img_num, 1);
predicted_gaze1 = reshape(svr_res_list, 15*img_num, 1);
predicted_gaze2 = reshape(lr_res_list, 15*img_num, 1);
predicted_gaze3 = reshape(knn_res_list, 15*img_num, 1);
images = reshape(img_list, 15*img_num, 1);
% convert cell array to ordinary array
real_gaze = cell2mat(real_gaze);
predicted_gaze1 = cell2mat(predicted_gaze1);
predicted_gaze2 = cell2mat(predicted_gaze2);
predicted_gaze3 = cell2mat(predicted_gaze3);
imgs = zeros(36, 60, length(images));
for i = 1:length(images)
    imgs(:, :, i) = images{i};
end
draw_gaze(real_gaze, predicted_gaze1, predicted_gaze2, predicted_gaze3, imgs);

[m, n] = size(imgs(:, :, 1));
for i = 1:length(images)
    img = imread(strcat('result/', num2str(i), '.png'));
    img = imresize(img, [m*3, n*3]);
    imwrite(img, strcat('result/', num2str(i), '.png'));
end

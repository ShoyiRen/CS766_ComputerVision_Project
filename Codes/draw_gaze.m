function draw_gaze(real_gaze, predicted_gaze1, predicted_gaze2, predicted_gaze3, img)
% convert sphere coordinates into x-y-z coordinates: real gaze
y_real = -sin(real_gaze(:, 1));
temp1 = tan(real_gaze(:, 2));
z_real = sqrt((1 - y_real.^2) ./ (temp1.^2 + 1));
x_real = temp1 .* z_real;
% update real gaze
real_gaze = [x_real, y_real, z_real];
% convert sphere coordinates into x-y-z coordinates: predicted gaze
y_predicted = -sin(predicted_gaze1(:, 1));
temp2 = tan(predicted_gaze1(:, 2));
z_predicted = sqrt((1 - y_predicted.^2) ./ (temp2.^2 + 1));
x_predicted = temp2 .* z_predicted;
% update predicted gaze
predicted_gaze1 = [x_predicted, y_predicted, z_predicted];
% convert sphere coordinates into x-y-z coordinates: predicted gaze
y_predicted = -sin(predicted_gaze2(:, 1));
temp2 = tan(predicted_gaze2(:, 2));
z_predicted = sqrt((1 - y_predicted.^2) ./ (temp2.^2 + 1));
x_predicted = temp2 .* z_predicted;
predicted_gaze2 = [x_predicted, y_predicted, z_predicted];
% convert sphere coordinates into x-y-z coordinates: predicted gaze
y_predicted = -sin(predicted_gaze3(:, 1));
temp2 = tan(predicted_gaze3(:, 2));
z_predicted = sqrt((1 - y_predicted.^2) ./ (temp2.^2 + 1));
x_predicted = temp2 .* z_predicted;
% update predicted gaze
predicted_gaze3 = [x_predicted, y_predicted, z_predicted];
% normalize gaze vector length to 50
real_gaze = real_gaze*80;
predicted_gaze1 = predicted_gaze1*80;
predicted_gaze2 = predicted_gaze2*80;
predicted_gaze3 = predicted_gaze3*80;

% plot gaze on img
[m, n, d] = size(img);
fh = figure();
for i = 1 : d
    cur_img = img(:, :, i);
    %cur_img = reshape(cur_img, 36, 60);
    [m, n] = size(cur_img);
    %imshow(uint8(cur_img));
    %iptsetpref('ImshowBorder','tight')
    imshow(uint8(cur_img), 'border', 'tight', 'initialmagnification', 'fit');
    % start positions: real gaze
    x1_real = round(n/2);
    y1_real = round(m/2);
    % start positions: predicted gaze
    x1_predicted = round(n/2);
    y1_predicted = round(m/2);
    % end positions: real gaze
    x2_real = x1_real + real_gaze(i, 1);
    y2_real = y1_real + real_gaze(i, 2);
    % end positions: predicted gaze
    x2_predicted = x1_predicted + predicted_gaze1(i, 1);
    y2_predicted = y1_predicted + predicted_gaze1(i, 2);
    x2_predicted_2 = x1_predicted + predicted_gaze2(i, 1);
    y2_predicted_2 = y1_predicted + predicted_gaze2(i, 2);
    x2_predicted_3 = x1_predicted + predicted_gaze3(i, 1);
    y2_predicted_3 = y1_predicted + predicted_gaze3(i, 2);
    % plot real gaze and predicted gaze on the img
    hold on;
    plot([x1_real, x2_real], [y1_real, y2_real], 'b', 'LineWidth', 4);
    plot([x1_predicted, x2_predicted], [y1_predicted, y2_predicted], 'r', 'LineWidth', 4);
    plot([x1_predicted, x2_predicted_2], [y1_predicted, y2_predicted_2], 'g', 'LineWidth', 4);
    plot([x1_predicted, x2_predicted_3], [y1_predicted, y2_predicted_3], 'y', 'LineWidth', 4);
    hold off;
    %annotated_img = saveAnnotatedImg(fh);
    out_file = strcat('result/', num2str(i), '.png');
    saveas(fh, out_file);
end
end

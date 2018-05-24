function main()
% Setup MEX environment
run_setup = 0;
if run_setup
    Setup();
end

%% Part 1
% Setup paths and constants
do_part_1 = 0;
pretrained_model = '../../data/models/fast-rcnn-caffenet-pascal07-dagnn.mat';
im_example = '../../example.jpg';
im_example_bbox = '../../example_boxes.mat';

% Preprocess network
raw_net = load(pretrained_model);
net = preprocessNet(raw_net);

% Process the single example image
im_single = imread(im_example);
bbox_single = load(im_example_bbox);
rois_single = single(bbox_single.boxes);
[im, rois] = resizeImBbox(single(im_single), rois_single);
im = bsxfun(@minus, im, net.meta.normalization.averageImage);

% Feed data into model
net.eval({'data', im, 'rois', rois'});
score = squeeze(net.getVar('cls_prob').value);
bbreg = squeeze(net.getVar('bbox_pred').value);

% Apply NMS to car detections
id_car = 8;
nms_threshold = 0.15;  % Max allowed IoU ratio
score_car_before_nms = score(id_car, :);
bbox_car_before_nms = bbox_transform_inv(rois_single, bbreg(1+(id_car-1)*4:id_car*4, :)');
cand = nms([bbox_car_before_nms, score_car_before_nms'], nms_threshold);
score_car = score_car_before_nms(cand);
bbox_car = bbox_car_before_nms(cand, :);

% Determine the best threshold
mn = min(score_car);
mx = max(score_car);
num_steps = 100;
steps = linspace(mn, mx, num_steps);
pos = repmat(score_car, num_steps, 1)' > repmat(steps, size(cand, 1), 1);
num_pos = sum(pos);
num_dt = 6;
id_chosen = find(num_pos == num_dt, 1, 'last');
part1_final_threshold = steps(id_chosen);
fprintf('Final threshold is %.2f.\n', part1_final_threshold);

% Visualize
if do_part_1
    if (~exist('../../pictures', 'dir'))
        mkdir('../../pictures');
    end
    plot(steps, num_pos);
    xlabel('Threshold');
    ylabel('Number of Detections');
    title('Number of Detections vs. Threshold');
    print(gcf, '-djpeg', '../../pictures/threshold_vs_numberOfDetection.jpg');
    close all;
    
    detections = find(pos(:, id_chosen))';
    imshow(im_single);
    hold on;
    for d = detections
        x = bbox_car(d, 1);
        y = bbox_car(d, 2);
        w = bbox_car(d, 3) - x + 1;
        h = bbox_car(d, 4) - y + 1;
        rectangle('Position', [x+1 y+1 w h], 'EdgeColor', 'r');
        str = sprintf('%0.2f', score_car(d));
        text(double(x+w/2+1), double(y+1), str,...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'bottom',...
            'BackgroundColor', 'r',...
            'Margin', 0.5, 'FontSize', 6);
    end
    print(gcf, '-djpeg', '../../pictures/car_detection.jpg');
    close all;
end

%% Part 2
% Setup paths and constants
get_results = 0;
ann_dir = '../../data/annotations';
img_dir = '../../data/images';
bbox_mat = '../../data/SSW/SelectiveSearchVOC2007test.mat';
rois_all = load(bbox_mat);
num_ims = 4952;

% Load annotations
ann_files = dir(ann_dir);
ann = cell(num_ims, 1);
cnt = 1;
for i = 1:numel(ann_files)
    f = ann_files(i);
    if (length(f.name) < 4)
        continue;
    end
    ann{cnt} = PASreadrecord(fullfile(ann_dir, f.name));
    cnt = cnt + 1;
end

% Run network on the entire testset
if get_results
    for i = 1:num_ims
        tic
        % Process each image
        im = imread(fullfile(img_dir, ann{i}.filename));
        [im, rois] = resizeImBbox(single(im), single(rois_all.boxes{i}));
        im = bsxfun(@minus, im, net.meta.normalization.averageImage);
        
        % Feed data into model
        net.eval({'data', im, 'rois', rois'});
        score = squeeze(net.getVar('cls_prob').value);
        bbreg = squeeze(net.getVar('bbox_pred').value);
        
        % Save the results
        results(i).name = ann{i}.filename;
        results(i).score = score;
        results(i).bbreg = bbreg;
        fprintf('Process image %d in %.2f seconds.\n', i, toc);
    end
    save('results.mat', '-v7.3', 'results');
else
    load('results.mat', 'results');
end

%% Merge results by class
% Setup constants
get_merge_results = 0;
num_cl = 20;
cl_top = 40;
total = cl_top * num_ims;

% Merge the results class-wise
if get_merge_results
    merge_results(num_cl) = struct('im_score_bbid', []);
    for c = 1:num_cl
        tic;
        for i = 1:num_ims
            score = results(i).score(c + 1, :);
            bbreg = results(i).bbreg(1+c*4:(c+1)*4, :);
            img_id = i * ones(1, size(score, 2));
            im_score_bbid_tuple = [img_id; score; 1:size(bbreg, 2)];
            merge_results(c).im_score_bbid =...
                [merge_results(c).im_score_bbid; im_score_bbid_tuple'];
        end
        % Sort detections based on score
        [~, I] = sort(merge_results(c).im_score_bbid(:, 2), 'descend');
        merge_results(c).im_score_bbid = merge_results(c).im_score_bbid(I(1:total), :);
        fprintf('Merge class %d in %.2f seconds.\n', c, toc);
    end
    save('merge_results.mat', '-v7.3', 'merge_results');
else
    load('merge_results.mat', 'merge_results');
end

%% Get a decent multi-classes detection
% Setup constants
best_num = 0;
best_im = 0;
best_bbox = 0;

% Find the best detection results among all images
for i = 1:num_ims
    curr_bbox = [];
    cl_include = 0;
    for c = 1:num_cl
        I = (merge_results(c).im_score_bbid(:, 1) == i) &...
            (merge_results(c).im_score_bbid(:, 2) > part1_final_threshold);
        dt_bbox_score = merge_results(c).im_score_bbid(I, 2);
        dt_bbox_id = merge_results(c).im_score_bbid(I, 3);
        bbox_cnt = numel(dt_bbox_id);
        if bbox_cnt > 0
            cl_include = cl_include + 1;
        end
        curr_bbox = [curr_bbox; [c * ones(bbox_cnt, 1), dt_bbox_id, dt_bbox_score]];
    end
    
    % Update the best detection
    if cl_include > best_num
        best_im = i;
        best_bbox = curr_bbox;
        best_num = cl_include;
    end
end

% Adjust bounding boxes based on bounding box regressions
final_bbox = zeros(size(best_bbox, 1), 6);
for k = 1:size(best_bbox, 1)
    bbox_cl = best_bbox(k, 1);
    bbox_id = best_bbox(k, 2);
    bbox_score = best_bbox(k, 3);
    bbreg = results(best_im).bbreg(1+bbox_cl*4:(bbox_cl+1)*4, bbox_id);
    score = results(best_im).score(bbox_cl + 1, bbox_id);
    bbox_cand = rois_all.boxes{best_im}(bbox_id, :);
    final_bbox(k, :) = [bbox_cl bbox_transform_inv(bbox_cand, bbreg') bbox_score'];
end

% Apply NMS to all boxes detected
unique_cl = unique(final_bbox(:, 1));
final_dt = [];
for j = 1:numel(unique_cl)
    c = unique_cl(j);
    curr_cl = final_bbox(final_bbox(:, 1) == c, 2:end);
    cand = nms(curr_cl, nms_threshold);
    final_dt = [final_dt; [c * ones(size(cand, 1), 1) curr_cl(cand, :)]];
end

% Visualize
colors = 'ycmrgbycmrgbycmrgbycmrgb';
figure;
imshow(fullfile(img_dir, ann{best_im}.filename));
hold on;
for k = 1:size(final_dt)
    bbox = final_dt(k, :);
    cl_label = net.meta.classes.name{bbox(1) + 1};
    x = bbox(2);
    y = bbox(3);
    w = bbox(4) - x + 1;
    h = bbox(5) - y + 1;
    score = bbox(6);
    color = colors(bbox(1));
    rectangle('Position', [x+1 y+1 w h], 'EdgeColor', color);
    str = sprintf('%s: %0.2f', cl_label, score);
    text(double(x+w/2+1), double(y+1), str,...
            'HorizontalAlignment', 'center',...
            'VerticalAlignment', 'bottom',...
            'BackgroundColor', color,...
            'Margin', 0.5, 'FontSize', 6);
end
print(gcf, '-djpeg', '../../pictures/best_detection.jpg');
close all;

% Compute AP of every category and mAP
mAP = 0;
im_top = 100;
class_results(num_cl) = struct('dts', [],...
                               'gts', [],...
                               'precision', [],...
                               'recall', [],...
                               'ap', []);
for c = 1:num_cl
    cl_label = net.meta.classes.name{c + 1};
    dts_all(num_ims) = struct('Boxes', [], 'Scores', []);
    for i = 1:num_ims
        I = (merge_results(c).im_score_bbid(:, 1) == i);
        score = merge_results(c).im_score_bbid(I, 2);
        bbox_id = merge_results(c).im_score_bbid(I, 3);
        if numel(bbox_id) ~= 0
            bbreg_all = results(i).bbreg(1+c*4:(c+1)*4, bbox_id);
            bbox_all = bbox_transform_inv(rois_all.boxes{i}(bbox_id, :),...
                bbreg_all');
            dt_chosen = nms([bbox_all score], nms_threshold);
            num_dt_chosen = min(im_top, size(dt_chosen, 1));
            bbreg = bbox_all(dt_chosen(1:num_dt_chosen), :);

            dts_all(i).Boxes = xyxy2xywh(bbreg) + [1 1 0 0];
            dts_all(i).Scores = score(dt_chosen(1:num_dt_chosen));
        else
            dts_all(i).Boxes = [];
            dts_all(i).Scores = [];
        end
        
        bbox_gt = [];
        for obj = ann{i}.objects
            if strcmp(obj.class, cl_label)
                xywh_gt = xyxy2xywh(obj.bbox) + [1 1 0 0];
                bbox_gt = [bbox_gt; xywh_gt];
            end
        end
        gts_all(i).Boxes = bbox_gt;
    end
    
    [ap, recall, precision] = evaluateDetectionPrecision(...
                                struct2table(dts_all),...
                                struct2table(gts_all), 0.5);
    class_results(c).dts = dts_all;
    class_results(c).gts = gts_all;
    class_results(c).ap = ap;
    class_results(c).recall = recall;
    class_results(c).precision = precision;
    fprintf('Class: %s, AP: %0.6f\n', cl_label, ap);
    mAP = mAP + ap;
end
save('class_results', '-v7.3', 'class_results');
mAP = mAP / num_cl;
fprintf('mAP: %0.6f\n', mAP);

%% Draw pr-curve for car class
figure;
plot(class_results(7).recall, class_results(7).precision);
xlabel('Recall Rate');
ylabel('Precision Rate');
title(sprintf('Precision-recall curve for car class, AP: %0.6f', class_results(7).ap));
print(gcf, '-djpeg', '../../pictures/pr_curve_car.jpg');
close all;
end

% Class: aeroplane, AP: 0.585358
% Class: bicycle, AP: 0.604260
% Class: bird, AP: 0.406702
% Class: boat, AP: 0.296687
% Class: bottle, AP: 0.145479
% Class: bus, AP: 0.615982
% Class: car, AP: 0.579787
% Class: cat, AP: 0.671733
% Class: chair, AP: 0.206817
% Class: cow, AP: 0.496006
% Class: diningtable, AP: 0.492875
% Class: dog, AP: 0.567848
% Class: horse, AP: 0.645948
% Class: motorbike, AP: 0.610507
% Class: person, AP: 0.477116
% Class: pottedplant, AP: 0.201672
% Class: sheep, AP: 0.415092
% Class: sofa, AP: 0.431161
% Class: train, AP: 0.649127
% Class: tvmonitor, AP: 0.520533
% mAP: 0.481034
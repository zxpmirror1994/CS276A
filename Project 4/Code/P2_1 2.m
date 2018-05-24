% prep
clear all;
close all;

% flags
flag_compile_libsvm_c = 0;
flag_compile_libsvm_mex = 0;

% compile libsvm
if flag_compile_libsvm_c
    parent = cd('libsvm-3.21');
    [status,cmdout] = system('make');
    cd(parent);
    disp(status);
    disp(cmdout);
end

if flag_compile_libsvm_mex
    parent = cd('libsvm-3.21/matlab');
    make;
    cd(parent);
end

% setup
diary('P2_1.out');
rng(123);
addpath('libsvm-3.21/matlab');

disp('loading data ...');
% load('P1_2.mat', 'paras');
load('hog/hog_sbin_32.mat', 'hog_elec_gov', 'hog_elec_sen');
hog32 = [hog_elec_gov; hog_elec_sen];
clear hog_elec_gov hog_elec_sen

load('stat-gov.mat');
gov_lm = face_landmark;
gov_diff = vote_diff;
load('stat-sen.mat');
sen_lm = face_landmark;
sen_diff = vote_diff;

face_landmark = [gov_lm; sen_lm];
features = [face_landmark hog32];
features = normalize_features(features);
sample_cnt = size(face_landmark, 1);
gt = [gov_diff; sen_diff];

disp('cross validation ...');
num_fold = 5;
acc = zeros(num_fold, 1);
acc_train = zeros(num_fold, 1);
prec = zeros(num_fold, 1);
prec_train = zeros(num_fold, 1);
paras = cell(num_fold, 1);

for i=1:num_fold
    test_sample_cnt = floor(sample_cnt/num_fold);
    first = (i - 1) * test_sample_cnt + 1;
    last = i * test_sample_cnt;
    
    test_feat = features(first:last,:);
    test_gt = gt(first:last);

    train_feat = features([1:first-1 last+1:end], :);
    train_gt = gt([1:first-1 last+1:end], :);
    
    [c, gamma, eps] = find_best_paras(train_gt, train_feat);
    paravec = sprintf('-s 3 -c %f, -g %f, -p %f', 2^c, 2^gamma, 2^eps);
    paras{i} = svmtrain(train_gt, train_feat, paravec);
    
    [res,~,~] = svmpredict(test_gt, test_feat, paras{i}, '-q');
    acc(i) = sum((res>=0)==(test_gt>=0)) / size(test_gt,1);
    tp = sum((res>=0).*(test_gt>=0));
    fp = sum((res>=0).*(test_gt<0));
    prec(i) = tp / (fp + tp);
    
    [res,~,~] = svmpredict(train_gt, train_feat, paras{i}, '-q');
    acc_train(i) = sum((res>=0)==(train_gt>=0)) / size(train_gt,1);
    tp=sum((res>=0).*(train_gt>=0));
    fp=sum((res>=0).*(train_gt<0));
    prec_train(i) = tp / (fp + tp); 
end

avg_acc = mean(acc,1);
avg_acc_train = mean(acc_train,1);

avg_prec = mean(prec,1);
avg_prec_train = mean(prec_train,1);

save('P2_1.mat','paras','avg_prec_train','avg_prec','avg_acc_train',...
    'avg_acc');

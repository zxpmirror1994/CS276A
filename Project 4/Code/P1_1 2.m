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
diary('P1_1.out');
rng(123);
addpath('libsvm-3.21/matlab');

% data
disp('loading data ...');
load('train-anno.mat', 'face_landmark', 'trait_annotation');
features = face_landmark;
labels = trait_annotation;
sample_cnt = size(face_landmark, 1);
gt = double(trait_annotation >= repmat(mean(trait_annotation, 1),...
    [size(trait_annotation, 1),1]));

% scale
face_landmark = normalize_features(face_landmark);
face_landmark = face_landmark(:, [1:76 81:156]);

% predict
disp('cross validation ...');
num_fold = 5;
num_class = 14;
acc = zeros(num_fold, num_class);
acc_train = zeros(num_fold, num_class);
prec = zeros(num_fold, num_class);
prec_train = zeros(num_fold, num_class);
paras = cell(num_fold, num_class);

for i=1:num_fold
    test_sample_cnt = floor(sample_cnt/num_fold);
    first = (i - 1) * test_sample_cnt + 1;
    last = i * test_sample_cnt;
    
    test_lm = face_landmark(first:last,:);
    test_gt = gt(first:last,:);
    
    train_lm = face_landmark([1:first-1 last+1:end], :);
    train_gt = gt([1:first-1 last+1:end], :);
    
    [c, gamma, eps] = find_best_paras(train_gt, train_lm);
    for j=1:num_class
        paravec = sprintf('-s 3 -c %f, -g %f, -p %f', 2^c(j), 2^gamma(j), 2^eps(j));
        paras{i,j} = svmtrain(train_gt(:,j), train_lm, paravec);
        
        [res,~,~] = svmpredict(test_gt(:,j), test_lm, paras{i,j}, '-q');
        acc(i,j) = sum(res==test_gt(:,j)) / size(test_gt,1);
        tp = sum((res==test_gt(:,j)).*(res==1));
        fp = sum((res~=test_gt(:,j)).*(res==1));
        prec(i,j) = tp / (fp + tp);
        
        [res,~,~] = svmpredict(train_gt(:,j), train_lm, paras{i,j}, '-q');
        acc_train(i,j) = sum(res==train_gt(:,j)) / size(train_gt,1);
        tp = sum((res==train_gt(:,j)).*(res==1));
        fp = sum((res~=train_gt(:,j)).*(res==1));
        prec_train(i,j) = tp / (fp + tp);
                
    end
end

avg_acc = mean(acc,1);
avg_acc_train = mean(acc_train,1);

avg_prec = mean(prec,1);
avg_prec_train = mean(prec_train,1);

save('P1_1.mat', 'paras','avg_prec_train','avg_prec','avg_acc_train','avg_acc');
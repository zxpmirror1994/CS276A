root = pwd();

cd(root);
cd('code/examples/');

mex -setup C++

run('setup.m')
vl_testnn();

% Task 1
run('cnn_cifar.m');

cd(root);
cd('code/examples/data/cifar-lenet/');

res = load('net-epoch-30.mat');
disp(res.info.train.error);
disp(res.info.val.error);

f = figure;
hold on;
plot(1:30,res.info.train.error);
plot(1:30,res.info.val.error);
xlabel('Epoch');
ylabel('Error');
title('Training and Validation Error (Full)');
legend('train', 'val');

cd(root);
cd('code/examples/');

copyfile model.mat model-full.mat;
copyfile data/cifar-lenet/net-epoch-30.mat data/cifar-lenet/net-epoch-30-full.mat;

% Task 2
open('cnn_cifar_init.m');
% Modify the block 5 filter size
run('cnn_cifar.m');

cd(root);
cd('code/examples/data/cifar-lenet/');

res = load('net-epoch-30.mat');
disp(res.info.train.error);
disp(res.info.val.error);

% Task 3
cd(root);
run(fullfile(root, 'code', 'matlab', 'vl_setupnn.m'));

cd(root);
cd('code/examples/');
model = load('model-full.mat');
model.net.layers = model.net.layers(1:end-1); % remove softmax

for i = 1:10
    img = imread(sprintf('images/%d.png',i));
    img = single(img) - model.net.averageImage;
    res = vl_simplenn(model.net, img);
    responses = res(2).x;

    f = figure;
    for j = 1:32
        subplot(4,8,j);
        image(responses(:,:,j));
        axis tight;
        axis off;
        daspect([1 1 1]);
    end
    
    saveas(f, sprintf('images/%d_filter_res.png',i));
end

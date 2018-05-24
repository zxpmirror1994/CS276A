function [] = P1_4(im_train, lm_train)
    if (~exist('pictures', 'dir'))
        mkdir('pictures');
    end
    if (~exist('pictures/P1_4', 'dir'))
        mkdir('pictures/P1_4');
    end
    
    x = 256;
    y = 256;
    
    n_train = size(im_train, 1);
    im_size = size(im_train, 2);
    lm_size = size(lm_train, 2);
    first_n = 10;
    rf_num = 20;
    
    % Apply PCA on the warpped training images
    [lm_mean, V_lm, S_lm] = PCA(lm_train);
    im_train_warp = warpImage(im_train, lm_train, ones(n_train, 1) * lm_mean);
    [im_mean_warp, V_warp, S_warp] = PCA(im_train_warp);
    
    % Synthesize 20 faces using random sampling of
    % eigenfaces/eigen-warppings
    im_rand = zeros(rf_num, im_size);
    lm_rand = zeros(rf_num, lm_size);
    
    for i = 1:rf_num
        for j = 1:first_n
            im_rand_ev = normrnd(0.0, 1.0) * sqrt(S_warp(j,j)/n_train);
            lm_rand_ev = normrnd(0.0, 1.0) * sqrt(S_lm(j,j)/n_train);
            im_rand(i,:) = im_rand(i,:) + im_rand_ev * V_warp(j,:);
            lm_rand(i,:) = lm_rand(i,:) + lm_rand_ev * V_lm(j,:);
        end
        im_rand(i,:) = im_rand(i,:) + im_mean_warp;
        lm_rand(i,:) = lm_rand(i,:) + lm_mean;
    end
    
    im_rand_warp = warpImage(im_rand, ones(rf_num, 1) * lm_mean, lm_rand);
    figure();
    for i = 1:rf_num
        subplot(4,5,i);
        imshow(reshape(uint8(im_rand_warp(i,:)), x, y));
    end
    print(gcf, '-djpeg', './pictures/P1_4/synthesized_face.jpg');
    close all;
end
function [] = P1_3(im_train, im_test, lm_train, lm_test) 
    if (~exist('pictures', 'dir'))
        mkdir('pictures');
    end
    if (~exist('pictures/P1_3', 'dir'))
        mkdir('pictures/P1_3');
    end
    
    x = 256;
    y = 256;
    
    n_train = size(im_train, 1);
    n_test = size(im_test, 1);
    im_size = size(im_test, 2);
    first_n = 10;
    
    % Apply PCA on the warpped training images
    [lm_mean, V_lm] = PCA(lm_train);
    im_train_warp = warpImage(im_train, lm_train, ones(n_train, 1) * lm_mean);
    [im_mean_warp, V_warp] = PCA(im_train_warp);  
    
    figure();
    for i = 1:first_n
        subplot(3,4,i);
        V_warp_norm = 255 * mat2gray(reshape(V_warp(i,:),x,y));
        imshow(uint8(V_warp_norm));
        title(['warpped\_ef\_id = ' num2str(i)]);
    end
    print(gcf, '-djpeg', './pictures/P1_3/first_10_warpped_eigenfaces.jpg');
    close all;
    
    % Reconstruct the testing landmarks and warp the testing images
    lm_rec = REC(lm_test, lm_mean, V_lm, first_n) ;
    im_test_warp = warpImage(im_test, lm_rec, ones(n_test, 1) * lm_mean);
    
    % Reconstruct the testing images at mean position/warpped position
    im_rec_mean_pos = REC(im_test_warp, im_mean_warp, V_warp, first_n);
    im_rec_warp = warpImage(im_rec_mean_pos, ones(n_test, 1) * lm_mean, lm_rec);

    figure();
    for i = 1:n_test
        subplot(5,6,i);
        imshow(reshape(uint8(im_rec_mean_pos(i,:)),x,y));
    end
    print(gcf, '-djpeg', './pictures/P1_3/reconstruction_mean_pos.jpg');
    close all;
    
    figure();
    for i = 1:n_test
        subplot(5,6,i);
        imshow(reshape(uint8(im_rec_warp(i,:)),x,y));
    end
    print(gcf, '-djpeg', './pictures/P1_3/reconstruction_warpped.jpg');
    close all;
    
    % Calculate the reconstruction error
    warp_error = zeros(n_train,1);
    for ef_num = 1:n_train
        im_test_warp = warpImage(im_test, lm_test, ones(n_test, 1) * lm_mean);
        im_rec_mean_pos = REC(im_test_warp, im_mean_warp, V_warp, ef_num);
        im_rec_warp = warpImage(im_rec_mean_pos, ones(n_test, 1) * lm_mean, lm_rec);
        warp_error(ef_num) = sum(sum((im_test - im_rec_warp).^2)) / (n_test * im_size);
    end
    
    figure()
    plot(warp_error);
    title('Reconstruction Error for Testing Images (Geometry + Appearance)');
    xlabel('# of Eigen-faces Used');
    ylabel('Reconstruction Error');
    print(gcf, '-djpeg', './pictures/P1_3/reconstruction_error.jpg');
    close all;
end
    
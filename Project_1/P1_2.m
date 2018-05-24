function [] = P1_2(im_train, im_test, lm_train, lm_test)
    if (~exist('pictures', 'dir'))
        mkdir('pictures');
    end
    if (~exist('pictures/P1_2', 'dir'))
        mkdir('pictures/P1_2');
    end
    
    x = 256;
    y = 256;
    lm_num = 87;
    
    % Apply PCA on both images and landmarks
    [lm_mean, V] = PCA(lm_train);
    [im_mean, ~] = PCA(im_train);
    
    % Display the mean landmark on the mean face
    figure();
    imshow(reshape(uint8(im_mean), x, y));
    hold on;
    plot(lm_mean(1:lm_num), lm_mean(lm_num+1:lm_num*2), 'r.', 'MarkerSize', 5);
    hold off;
    print(gcf, '-djpeg', './pictures/P1_2/mean_landmark.jpg');
    close all;
    
    % Find the first 5 eigen-warppings
    first_n = 5;
    figure();
    for i = 1:first_n
        subplot(2,3,i);
        V_display = V(i,:) .* 10 + lm_mean;
        plot(V_display(1:lm_num), V_display(lm_num+1:lm_num*2), 'r.', 'MarkerSize', 5);
        axis([0 256 0 256]);
        set(gca , 'Ydir', 'reverse');
        title(['ew\_id = ' num2str(i)]);
    end
    print(gcf, '-djpeg', './pictures/P1_2/eigen_warppings.jpg');
    close all;
    
    % Reconstruct the testing landmarks using the first 5 eigen-warppings
    lm_rec = REC(lm_test, lm_mean, V, first_n);
    for i = 1:size(im_test, 1)
        new_i = mod(i-1, 9);
        if (new_i == 0) 
            figure();
        end
        subplot(3,3,new_i+1);
        imshow(reshape(uint8(im_test(i,:)), x, y));
        hold on;
        plot(lm_rec(i,1:lm_num), lm_rec(i,lm_num+1:lm_num*2), 'r.', 'MarkerSize', 5);
        plot(lm_test(i,1:lm_num), lm_test(i,lm_num+1:lm_num*2), 'g.', 'MarkerSize', 5);
        set(gca , 'Ydir','reverse');
        title(['test\_id = ' num2str(i)]);
        if (new_i == 8) 
            print(gcf, '-djpeg', sprintf('./pictures/P1_2/test_reconstruction_set%d.jpg', ceil(i/9)));
        end
    end
    close all;
    
    % Calculate the reconstruction error 
    figure()
    lm_err = REC_Error(lm_test, lm_mean, V, 'P1_2');
    plot(lm_err);
    title('Reconstruction Error for Testing Landmarks');
    xlabel('# of Eigen-warppings Used');
    ylabel('Reconstruction Error');
    print(gcf, '-djpeg', './pictures/P1_2/test_reconstruction_error.jpg');
    close all;
end
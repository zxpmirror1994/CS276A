function [] = P1_1(im_train, im_test)
    if (~exist('pictures', 'dir'))
        mkdir('pictures');
    end
    if (~exist('pictures/P1_1', 'dir'))
        mkdir('pictures/P1_1');
    end
    
    x = 256;
    y = 256;
    
    % Find the mean face and normalized eigenfaces
    [im_mean, V] = PCA(im_train);
    
    figure();
    imshow(reshape(uint8(im_mean), x, y));
    print(gcf, '-djpeg', './pictures/P1_1/mean_face.jpg');
    close all;
    
    % Display the first 20 eigenfaces
    first_n = 20;
    figure();
    for i = 1:first_n
        subplot(4,5,i);
        V_norm = 255 * mat2gray(reshape(V(i,:),x,y));
        imshow(uint8(V_norm));
        title(['ef\_id = ' num2str(i)]);
    end
    print(gcf, '-djpeg', './pictures/P1_1/first_20_eigenfaces.jpg');
    close all;
    
    % Reconstruct the test images using the first 20 eigenfaces
    im_rec = REC(im_test, im_mean, V, first_n);
    figure()
    for i = 1:size(im_rec, 1)
        subplot(5,6,i);
        imshow(reshape(uint8(im_rec(i, :)), x, y));
        title(['test\_id = ' num2str(i)]);
    end
    print(gcf, '-djpeg', './pictures/P1_1/test_reconstruction.jpg');
    close all;
    
    % Calculate the reconstruction error
    im_err = REC_Error(im_test, im_mean, V, 'P1_1');
    figure()
    plot(im_err);
    title('Reconstruction Error for Testing Images');
    xlabel('# of Eigenfaces Used');
    ylabel('Reconstruction Error');
    print(gcf, '-djpeg', './pictures/P1_1/test_reconstruction_error.jpg');
    close all;
end
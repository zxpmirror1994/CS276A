function [] = P2_2(im_male, im_female, lm_male, lm_female)
    if (~exist('pictures', 'dir'))
        mkdir('pictures');
    end
    if (~exist('pictures/P2_2', 'dir'))
        mkdir('pictures/P2_2');
    end
    
    % Testing images(landmarks): first 10 male images(landmarks) + first 10 female images(landmarks)
    % The rest are put into the training image(landmark) set
    im_test = [im_male(1:10, :); im_female(1:10, :)];
    lm_test = [lm_male(1:10, :); lm_female(1:10, :)];
    im_male = im_male(11:end, :);
    im_female = im_female(11:end, :);
    lm_male = lm_male(11:end, :);
    lm_female = lm_female(11:end, :);
    
    for new_dim=[10 50]
        % Landmark Data preparation
        lm_all = [lm_male; lm_female];
        [lm_mean, V_lm] = PCA(lm_all);
        
        lm_male_minus_mean = lm_male - lm_mean;
        lm_female_minus_mean = lm_female - lm_mean;
        lm_test_minus_mean = lm_test - lm_mean;
        
        % Perform PCA to reduce the landmark data dimension
        lm_male_proj = lm_male_minus_mean * V_lm(1:new_dim,:)';
        lm_female_proj = lm_female_minus_mean * V_lm(1:new_dim,:)';
        lm_test_proj = lm_test_minus_mean * V_lm(1:new_dim,:)';
        
        % Low-dimensional Fisherface:
        lm_w = LowDim_Fisher(lm_male_proj, lm_female_proj);
        
        % Image Data preparation
        im_all = [im_male; im_female];
        [im_mean, V_im] = PCA(im_all);
        
        im_male_minus_mean = im_male - im_mean;
        im_female_minus_mean = im_female - im_mean;
        im_test_minus_mean = im_test - im_mean;
        
        % Perform PCA to reduce the image data dimension
        im_male_proj = im_male_minus_mean * V_im(1:new_dim,:)';
        im_female_proj = im_female_minus_mean * V_im(1:new_dim,:)';
        im_test_proj = im_test_minus_mean * V_im(1:new_dim,:)';
        
        % Low-dimensional Fisherface:
        im_w = LowDim_Fisher(im_male_proj, im_female_proj);
        
        % Plot
        figure();
        hold on;
        for i = 1:size(im_male)
            im_male_fisher(i) = im_w' * im_male_proj(i,:)';
            lm_male_fisher(i) = lm_w' * lm_male_proj(i,:)';
        end
        plot(im_male_fisher, lm_male_fisher, 'b+');
        for i = 1:size(im_female)
            im_female_fisher(i) = im_w' * im_female_proj(i,:)';
            lm_female_fisher(i) = lm_w' * lm_female_proj(i,:)';  
        end
        plot(im_female_fisher, lm_female_fisher, 'r+');
        for i = 1:10
            im_male_test_fisher(i) = im_w' * im_test_proj(i,:)';
            lm_male_test_fisher(i) = lm_w' * lm_test_proj(i,:)';
        end
        plot(im_male_test_fisher, lm_male_test_fisher, 'bo');
        for i = 11:20
            im_female_test_fisher(i) = im_w' * im_test_proj(i,:)';
            lm_female_test_fisher(i) = lm_w' * lm_test_proj(i,:)';
        end
        plot(im_female_test_fisher, lm_female_test_fisher, 'ro');
        title(sprintf('2D-Projected Fisher Face Differentiation (dim = %d)', new_dim));
        xlabel('Image Projection');
        ylabel('Landmark Projection');
        legend('training: male', 'training: female', 'testing: male', 'testing: female');
        print(gcf, '-djpeg', sprintf('./pictures/P2_2/2D_fisher_%d.jpg', new_dim));
        close all;
    end
end
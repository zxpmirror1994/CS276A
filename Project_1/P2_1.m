function [] = P2_1(im_male, im_female)
    if (~exist('pictures', 'dir'))
        mkdir('pictures');
    end
    if (~exist('pictures/P2_1', 'dir'))
        mkdir('pictures/P2_1');
    end
    
    % Testing images: first 10 male images + first 10 female images
    % The rest are put into the training image set
    im_test = [im_male(1:10,:); im_female(1:10,:)];
    im_male = im_male(11:end,:);
    im_female = im_female(11:end,:);
  
    % High-dimensional Fisherface:
    high_w = HighDim_Fisher(im_male, im_female);
    
    % Plot
    FisherPlot(high_w, im_male, im_female, im_test(1:10, :), im_test(11:20, :));
    title('Fisher Face Differentiation');
    xlabel('Data Index');
    ylabel('Image Projection');
    legend('training: male', 'training: female', 'testing: male', 'testing: female', 'boundary');
    print(gcf, '-djpeg', './pictures/P2_1/high_dim_fisher.jpg');
    close all;
    
    % Data preparation
    im_all = [im_male; im_female];
    [im_mean, V] = PCA(im_all);
    im_male_minus_mean = im_male - im_mean;
    im_female_minus_mean = im_female - im_mean;
    im_test_minus_mean = im_test - im_mean;
    
    for new_dim=[10 50]
        % Perform PCA to reduce the data dimension
        im_male_proj = im_male_minus_mean * V(1:new_dim,:)';
        im_female_proj = im_female_minus_mean * V(1:new_dim,:)';
        im_test_proj = im_test_minus_mean * V(1:new_dim,:)';
        
        % Low-dimensional Fisherface:
        low_w = LowDim_Fisher(im_male_proj, im_female_proj);
        
        % Plot
        FisherPlot(low_w, im_male_proj, im_female_proj, im_test_proj(1:10, :), im_test_proj(11:20, :));
        title(sprintf('Reduced Dimension Fisher Face Differentiation (dim = %d)', new_dim));
        xlabel('Data Index');
        ylabel('Image Projection');
        legend('training: male', 'training: female', 'testing: male', 'testing: female', 'boundary');
        print(gcf, '-djpeg', sprintf('./pictures/P2_1/low_dim_fisher_%d.jpg', new_dim));
        close all;
    end
end
function [ ] = FisherPlot(w, im_male_train, im_female_train, im_male_test, im_female_test)
    % Plot the datapoints as well as the boundary separating them
    mean_1 = mean(im_male_train);
    mean_2 = mean(im_female_train);
    bd = w' * (mean_1' + mean_2') / 2.0;
    
    figure();
    hold on;
    % Male training
    for i = 1:size(im_male_train, 1)
        male_fisher(i) = w' * im_male_train(i, :)';
    end
    plot(male_fisher, 'b+');
    
    % Female training
    for i = 1:size(im_female_train, 1)
        female_fisher(i) = w' * im_female_train(i, :)';
    end
    plot(female_fisher, 'r+');
    
    % Male testing
    for i = 1:size(im_male_test, 1)
        male_fisher_test(i) = w' * im_male_test(i, :)';
    end
    plot(male_fisher_test, 'bo');
    
    % Female testing
    for i = 1:size(im_female_test, 1)
        female_fisher_test(i) = w' * im_female_test(i, :)';
    end
    plot(female_fisher_test, 'ro');
    
    % Boundary
    plot(ones(1, 80) * (bd), 'yellow');
end
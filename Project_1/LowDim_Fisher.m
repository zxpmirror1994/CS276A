function [low_w] = LowDim_Fisher(data_1, data_2)
    % Low-dimensional Fisherface
    % For more information, please refer to: compute the Fisher Face.pdf
    mean_1 = mean(data_1);
    mean_2 = mean(data_2);
    n_1 = size(data_1, 1);
    n_2 = size(data_2, 1);
    
    scat_1 = zeros(size(data_1, 2));
    scat_2 = zeros(size(data_2, 2));
    for i = 1:n_1
        res_1 = data_1(i, :) - mean_1;
        scat_1 = scat_1 + res_1' * res_1;
    end
    for i = 1:n_2
        res_2 = data_2(i, :) - mean_2;
        scat_2 = scat_2 + res_2' * res_2;
    end
    
    low_w = (scat_1 + scat_2) \ (mean_1 - mean_2)';
    low_w = low_w ./ norm(low_w);
end
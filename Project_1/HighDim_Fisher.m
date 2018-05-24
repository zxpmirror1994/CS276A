function [high_w] = HighDim_Fisher(data_1, data_2) 
    % High-dimensional Fisherface
    % For more information, please refer to: compute the Fisher Face.pdf
    data_size = size(data_1, 2);
    mean_1 = mean(data_1);
    mean_2 = mean(data_2);
    n_1 = size(data_1, 1);
    n_2 = size(data_2, 1);
    
    C = [data_1; data_2]';
    [~, S, V] = svd(C' * C);
    A = zeros(data_size, (n_1 + n_2));
    for i = 1:(n_1 + n_2)
        temp = C * V(:, i);
        A(:, i) = temp .* S(i, i) ./ norm(temp)^2;
    end
    y = A' * (mean_1 - mean_2)';
    high_w = C * ((S^2 * V') \ y);
    high_w = high_w ./ norm(high_w);
end
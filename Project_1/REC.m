function [x_rec] = REC(x_test, x_mean, V, first_n)
    % Vectorized form of reconstruction
    x_test_minus_mean = x_test - x_mean;
    x_rec = x_test_minus_mean * V(1:first_n,:)' * V(1:first_n,:);
    x_rec = x_rec + x_mean; 
end
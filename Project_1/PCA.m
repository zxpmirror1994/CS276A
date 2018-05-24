function [x_mean, V, S] = PCA(x_train)
    x_mean = mean(x_train);
    n_train = size(x_train, 1);
    
    % Find the eigenvectors and the eigenvalues of the training images/landmarks
    % with the mean subtracted
    x_train_minus_mean = x_train - x_mean;
    [U, S] = svd(x_train_minus_mean * x_train_minus_mean');
    V = [];
    for i = 1:n_train
        V(i,:) = x_train_minus_mean' * U(:,i);
        V(i,:) = V(i,:) / norm(V(i,:)); % Normalization
    end
end
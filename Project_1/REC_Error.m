function [x_err] = REC_Error(x_test, x_mean, V, id)
    ev_size = size(V,1);
    x_err = zeros(ev_size,1);
    
    for i = 1:ev_size
        x_rec = REC(x_test, x_mean, V, i);
        if (strcmp(id, 'P1_1'))
            % Squared intensity difference
            x_err(i) = sum(sum((x_test - x_rec).^2)) / (size(x_test,1) * size(x_test,2));
        elseif (strcmp(id, 'P1_2'))
            % Distance
            x_square_diff = (x_test - x_rec).^2;
            x_dist = x_square_diff(:,1:size(V,2)/2) + x_square_diff(:,size(V,2)/2+1:size(V,2));
            x_err(i) = sum(sum(sqrt(x_dist))) / (size(x_test,1) * size(x_test,2) * 0.5);
        end
    end
end
function [im_warp] = warpImage(im_data, lm_data_old, lm_data_new)   
    % My warpedImg interface to warpImage_new
    x = 256;
    y = 256;
    lm_num = 87;
    
    for i = 1:size(im_data, 1)
        Image = reshape(im_data(i, :), x, y);
        originalMarks = reshape(lm_data_old(i, :), lm_num, 2);
        desiredMarks = reshape(lm_data_new(i, :), lm_num, 2);
        im_warp(i, :) = reshape(warpImage_new(Image, originalMarks, desiredMarks), 1, x * y);
    end
    im_warp = double(im_warp);
end
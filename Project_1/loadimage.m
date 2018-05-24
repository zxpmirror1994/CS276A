function [im_train, im_test, lm_train, lm_test] = loadimage()
    % Save the path
    root = pwd();
    cd ..;
    
    % Load the training and testing images (one image per row)
    im_data = [];
    cnt = 0;
    for i = 0:177
        if (i == 103) 
            continue; 
        end
        cnt = cnt + 1;
        im = imread(sprintf('./face_data/face/face%03d.bmp', i));
        im_data(cnt,:) = reshape(im, 1, size(im,1)*size(im,2));
    end
    
    im_train = im_data(1:150,:);
    im_test = im_data(151:177,:);
    
    % Load the training and testing landmarks (one image per row)
    lm_data = [];
    cnt = 0;
    
    for i = 0:177
        if (i == 103) 
            continue; 
        end
        cnt = cnt + 1;
        fid = fopen(sprintf('./face_data/landmark_87/face%03d_87pt.dat', i));
        fscanf(fid, '%d', 1);
        lm = fscanf(fid, '%f', [2,87])';
        lm_data(cnt,:) = reshape(lm, 1, 2*87);
        fclose(fid);
    end
    
    lm_train = lm_data(1:150, :);
    lm_test = lm_data(151:177, :);
    
    cd(root);
end
function [im_male, im_female, im_unknown, lm_male, lm_female, lm_unknown] = loadgender()
    % Save the path
    root = pwd();
    cd ..; 
    
    % Load the training and testing images (one image per row) of males
    im_male = [];
    cnt = 0;
    for i = 0:88
        if (i == 57) 
            continue; 
        end
        cnt = cnt + 1;
        im = imread(sprintf('./face_data/male_face/face%03d.bmp', i));
        im_male(cnt, :) = reshape(im, 1, size(im, 1) * size(im, 2));
    end
    
    % Load the training and testing landmarks (one image per row) of males
    lm_male = [];
    cnt = 0;
    for i = 0:88
        if (i == 57) 
            continue; 
        end
        cnt = cnt + 1;
        fid = fopen(sprintf('./face_data/male_landmark_87/face%03d_87pt.txt', i));
        lm = fscanf(fid, '%f', [2,87])';
        lm_male(cnt, :) = reshape(lm, 1, 2*87);
        fclose(fid);
    end
    
    % Load the training and testing images (one image per row) of females
    im_female = [];
    cnt = 0;
    for i = 0:84
        cnt = cnt + 1;
        im = imread(sprintf('./face_data/female_face/face%03d.bmp', i));
        im_female(cnt, :) = reshape(im, 1, size(im, 1) * size(im, 2));
    end 
    
    % Load the training and testing landmarks (one image per row) of females
    lm_female = [];
    cnt = 0;
    for i = 0:84
        cnt = cnt + 1;
        fid = fopen(sprintf('./face_data/female_landmark_87/face%03d_87pt.txt', i));
        lm = fscanf(fid, '%f', [2,87])';
        lm_female(cnt, :) = reshape(lm, 1, 2*87);
        fclose(fid);
    end
    
    % Load the training and testing images (one image per row) of unknowns
    im_unknown = [];
    cnt = 0;
    for i = 0:3
        cnt = cnt + 1;
        im = imread(sprintf('./face_data/unknown_face/face%03d.bmp', i));
        im_unknown(cnt, :) = reshape(im, 1, size(im, 1) * size(im, 2));
    end
    
    % Load the training and testing landmarks (one image per row) of unknowns
    lm_unknown = [];
    cnt = 0;
    for i = 0:3
        cnt = cnt + 1;
        fid = fopen(sprintf('./face_data/unknown_landmark_87/face%03d_87pt.txt', i));
        lm = fscanf(fid, '%f', [2,87])';
        lm_unknown(cnt, :) = reshape(lm, 1, 2*87);
        fclose(fid);
    end
    
    cd(root);
end
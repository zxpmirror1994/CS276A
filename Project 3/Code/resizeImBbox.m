function [im, rois] = resizeImBbox(im_before, rois_before)
    short_edge = 600;
    n = size(rois_before, 1);
    w = size(im_before, 1);
    h = size(im_before, 2);
    scale = short_edge/min(w, h);
    im = imresize(im_before, scale);
    rois = [ones(n, 1) rois_before * scale];
end
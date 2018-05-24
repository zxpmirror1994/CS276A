function [xywh] = xyxy2xywh(xyxy)
    xywh(:, 1) = xyxy(:, 1);
    xywh(:, 2) = xyxy(:, 2);
    xywh(:, 3) = xyxy(:, 3) - xyxy(:, 1) + 1;
    xywh(:, 4) = xyxy(:, 4) - xyxy(:, 2) + 1;
end
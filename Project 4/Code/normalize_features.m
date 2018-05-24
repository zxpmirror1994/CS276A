function res = normalize_features(face_lm)
    sample_cnt = size(face_lm, 1);
    lm_min = min(face_lm);
    lm_range = range(face_lm)+1e-8;
    res = (face_lm - repmat(lm_min, [sample_cnt, 1])) ./...
        repmat(lm_range, [sample_cnt, 1]);
end

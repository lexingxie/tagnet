%% setup X and R

tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore', 'target_tags');


% load data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'train_label', 'train_idx_map') 

feat_data_mat = fullfile(data_dir, 'Train_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(train_idx_map, imgid));

imglab = train_label(img_idx, found_wn) ;
tag_dim = tag81(found_wn);
ntag = length(tag_dim);

tagcnt = full(sum(imglab,2));
nlen = length(tagcnt);

% subselect some data
if ~isinf(num_data_sample)
    idx = rand_idx(nlen, 1e4) ;
else
    idx = 1 : nlen;
end

R = imglab(idx, :);
X = imgfeat(idx, :)';

for j = 1 : size(imglab, 2)
    out_flag = sample_pos_neg(imglab(:,j), neg_pos_ratio, max_num_pos, max_num_neg);
    R(out_flag, j) = R(out_flag, j)-.5 ;
    R(~out_flag, j) = 0;
    tmp = full(R(out_flag, j));
    fprintf(1, 'resample data for tag #%d "%s": %d positive, %d negative, %d discarded \n', ...
        j, tag_dim{j}, sum(tmp>0), sum(tmp<0), sum(~out_flag));
end


whos X U V R Y
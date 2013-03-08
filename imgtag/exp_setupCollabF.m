
vecshuf = inline('a(randperm(length(a)))');
%% setup X and R
tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore', 'target_tags');


% load data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'train_label', 'train_idx_map') 
% >> whos('-file', fullfile(data_dir, 'TrainTest_Label.mat'))
% Name                    Size                 Bytes  Class             Attributes
% 
%   tag1k                1000x1                 124242  cell                        
%   tag81                  81x1                   9972  cell                        
%   te_tag81           107859x81               1511696  double            sparse    
%   test_idx_map             -                     112  containers.Map              
%   test_label         107859x81               3238880  double            sparse    
%   test_tag_1k        107859x1000             9995816  double            sparse    
%   tr_tag81           161789x81               2249456  double            sparse    
%   train_idx_map            -                     112  containers.Map              
%   train_label        161789x81               4824000  double            sparse    
%   train_tag_1k       161789x1000            14971624  double            sparse

feat_data_mat = fullfile(data_dir, 'Train_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(train_idx_map, imgid));

imglab = train_label(img_idx, found_wn) ;
tag_dim = tag81(found_wn);
ntag = length(tag_dim);

tagcnt = full(sum(imglab,2));
nlen = length(tagcnt);

% find out which img is fit for collab filtering
i5 = find(tagcnt>= COLLAB_PREC_D);
n5 = length(i5) ;
% ntag_collab_f controls how many tags to keep 


% subselect some data
if ~isinf(num_data_sample) && num_data_sample>n5
    idx = rand_idx(n5, num_data_sample) ;
    idx = i5(idx);
else
    idx = i5(1 : n5);
end

R = imglab(idx, :);
Rt = imglab(idx, :); % test set for the collab filtering validation

[numimg, numtag] = size(R);

X = imgfeat(idx, :)';

% now sample some training tags per image
if ntag_collab_f < COLLAB_PREC_D
    for i = 1 : length(idx)
        jpos = find(R(i, :)>0); jneg = find(R(i, :)==0);
        jpos = vecshuf(jpos);   jneg = vecshuf(jneg); 
        
        train_ind = sort([jpos(1:ntag_collab_f), jneg(1:ntag_collab_f)]);
        test_ind = setdiff(1:numtag, train_ind);
        
        R(i, train_ind) = R(i, train_ind) - .5;
        R(i, test_ind) = 0;
        Rt(i, test_ind) = Rt(i, test_ind) - .5;
        Rt(i, train_ind) = 0;
    end
end

whos X U V R Y

%% learn the model
fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));
id_train = find(R(:));
[U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', max_iter, 'solver', 'lbfgs');

%% evaluate
% micro-AP
id_test = find(Rt(:));    
Re = X'*U'*V*Y;
p_all = compute_perf(Re(id_test), 1.*full(Rt(id_test)), 'store_raw_pr', 2);

fprintf(1, '%s micro-AP on %d test pairs: %0.4f, prior %0.4f\n', ...
    datestr(now,31), length(id_test), p_all.ap, p_all.prior );

% precision @ 5
topk = 5;
p5e = zeros(numimg, 1); % estimation
p5p = zeros(numimg, 1); % prior
for i = 1 : numimg
    jl = find(Rt(i, :)==.5); % positive groundtruth
    [~, jn] = sort(Re(i, Re(i, :)~=0),  'descend');
    %[~, jk] = sort(R5_k(i, :), 'descend');  % some baseline
    %[~, ju] = sort(Ru5(i, :), 'descend');
    
    p5e(i) = length(intersect(jn(1:topk), jl))/topk;
    p5p(i) = length(jl) / sum(Re(i, :)~=0) ;
    %tu5(i) = length(intersect(ju(1:topk), jl))/topk;
end

fprintf(1, '%s top%d precision on 5+ tags: %0.4f, prior %0.4f\n', ...
    datestr(now,31), topk, mean(p5e), mean(p5p) );


% sample by tag, replace with sample by image
%     for j = 1 : size(imglab, 2)
%         out_flag = sample_pos_neg(R(:, j), neg_pos_ratio, max_num_pos, max_num_neg);
%         R(out_flag, j) = R(out_flag, j)-.5 ;
%         R(~out_flag, j) = 0;
%         tmp = full(R(out_flag, j));
%         fprintf(1, 'resample data for tag #%d "%s": %d positive, %d negative, %d discarded \n', ...
%             j, tag_dim{j}, sum(tmp>0), sum(tmp<0), sum(~out_flag));
%     end



% SVM baseline, with no hyper par search

eval_str = 'eval_Testsvm2_' ;

exp_envsetup
exp_setparams

whos

if ~exist('num_data_sample', 'var')
    num_data_sample = inf;
end

disp(eval_str)
disp(num_data_sample)

%matlabpool local 6

%% load tag features, setup Y
tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore');

[vs, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
tag_feat = tag_feat(:, iv(1: NUMV));
Y = log(tag_feat + 1)';

%% setup X and R
% load data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'train_label', 'train_idx_map')

feat_data_mat = fullfile(data_dir, 'Train_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(train_idx_map, imgid));

imglab = train_label(img_idx, found_wn) ;
tag_dim = tag81(found_wn);

% subselect some data
ntag = size(imglab, 2) ;
tagcnt = full(sum(imglab,2));
nlen = length(tagcnt);

% subselect some data
if ~isinf(num_data_sample)
    idx = rand_idx(nlen, num_data_sample) ;
else
    idx = 1 : nlen;
end

R = imglab(idx, :);
X = imgfeat(idx, :)';

%% load entire test data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'test_label', 'test_idx_map') 
%... , 'test_label', 'test_idx_map');

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
%   train_tag_1k       161789x1000            14971624  double
%   sparse    

%% setup X
feat_data_mat = fullfile(data_dir, 'Test_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(test_idx_map, imgid));

imglab = test_label(img_idx, found_wn) ;

% use what is learned above
whos imgfeat imgid imglab

%% now compare classification
Xtest = imgfeat';
Rtest = zeros(size(imglab));

fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));

%gam = 0 ;
svm_param = cell(ntag, 1);
%parfor j = 1 : ntag
for j = 1 : ntag
    fprintf(1, 'Learning linear svm for tag #%d, "%s"\n', j, tag_dim{j});
    % linear SVM baseline
    %[mod_final, ~] = cv_svm_wrapper2(1.*imglab(:,j), X', ...
    %    'do_normalize', true, 'kernel_type', 0, 'gam', 0, 'Cpen', 10.^(-1:3), 'neg_pos_ratio', 8);
    svm_options = '-t 0 -c 1 -q';
    train_label = 1.*(R(:,j)~=0) - 1.*(R(:,j)==0)  ;
    
    out_flag = sample_pos_neg(train_label, neg_pos_ratio, max_num_pos, max_num_neg);
    fprintf(1, ' resample data : %d positive, %d negative, %d discarded \n', ...
        sum(train_label(out_flag)>0), sum(train_label(out_flag)<0), sum(~out_flag));
    
    train_data = X(:, out_flag)' ;
    lsvm = svmtrain(train_label(out_flag), train_data, svm_options);
    svm_param{j} = lsvm;
    
    [~, ~, Rtest(:, j)] = svmpredict(imglab(:,j), Xtest', svm_param{j});
    pk = compute_perf(Rtest(:, j), 1.*full(imglab(:,j)), 'store_raw_pr', 2);
    
    fprintf(1, '%s tag%d "%s" test ap: %0.4f, auc: %0.4f, F1: %0.4f\n\n', datestr(now, 31), j, tag_dim{j}, pk.ap, pk.auc, pk.f1);
    
end
save(sav_file, 'svm_param', 'tag_dim') ;

%Rtest = Xtest'*U'*V*Y;
p_all = compute_perf(Rtest(:), 1.*full(imglab(:)), 'store_raw_pr', 2);

% compute KNN baseline
[~, n] = size(Xtest);
norm_X = sum(X.^2, 1);  % 1 x n vector
Rtest_k = zeros(size(Rtest));
parfor i = 1 : n
    dx = -2*Xtest(:, i)'*X + norm_X;
    [~, ix] = sort(dx);
    Rtest_k(i, :) = sum(R(ix(1:K), :), 1);
end
pktest = compute_perf(Rtest_k(:), 1.*full(imglab(:)), 'store_raw_pr', 2);

fprintf(1, ' AP on ALL-TEST matchbox %0.4f, with knn %0.4f, prior %0.4f\n', p_all.ap, pktest.ap, p_all.prior);

% compute per=concept AP for both
ap_concept = zeros(ntag, 2);
prior_concept = zeros(ntag, 1);
for i = 1 : ntag
    p = compute_perf(Rtest(:,i), 1.*full(imglab(:,i)), 'store_raw_pr', 2);
    ap_concept(i,1) = p.ap;
    prior_concept(i) = p.prior;
    pk = compute_perf(Rtest_k(:,i), 1.*full(imglab(:,i)), 'store_raw_pr', 2);
    ap_concept(i,2) = pk.ap;
end
fprintf(1, ' MAP on allTest: \n\tmatchbox %0.4f, knn %0.4f, prior %0.4f\n',...
    mean(ap_concept), mean(prior_concept));

[~, ik] = sort( prior_concept, 'descend' );
for i = 1 : ntag
    jj = ik(i);
    fprintf(1, '%0.4f\t %0.4f\t %0.4f\t %s \n', ap_concept(jj, :), prior_concept(jj), tag_dim{jj});
end

clear imgfeat imgid imglab X* R
save(sav_file) 

matlabpool close

diary off

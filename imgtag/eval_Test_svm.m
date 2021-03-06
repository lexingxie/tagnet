

eval_str = 'eval_Testsvm_' ;

exp_envsetup
exp_setparams

whos

matlabpool local 8

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

X = imgfeat'; %([id1;id2], :)';

%save(sav_file) 

fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));

%gam = 0 ;
svm_param = cell(ntag, 1);
parfor j = 1 : ntag
    fprintf(1, 'Learning linear svm for tag #%d, "%s"\n', j, tag_dim{j});
    % linear SVM baseline
    curid = (imglab(:,j)~=0) ;
    [mod_final, ~] = cv_svm_wrapper2(1.*imglab(:,j), X', ...
        'do_normalize', true, 'kernel_type', 0, 'gam', 0, 'Cpen', 10.^(-1:3), 'neg_pos_ratio', 8);
    svm_param{j} = mod_final.svm;
    
end

%% evaluate on Test set

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
%   train_tag_1k       161789x1000            14971624  double            sparse    


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
parfor j = 1 : ntag
    % evaluate
    [~, ~, Rtest(:, j)] = svmpredict(imglab(:,j), Xtest', svm_param{j});
    %pk = compute_perf(Rtest(:, j), 1.*full(imglab(:,j)), 'store_raw_pr', 2);
end
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

disp(tag_dim)
disp([ap_concept, prior_concept] )

save(sav_file) 

matlabpool close

diary off

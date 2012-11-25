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


%% setup X and Rtest
feat_data_mat = fullfile(data_dir, 'Test_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(test_idx_map, imgid));

imglab = test_label(img_idx, found_wn) ;


% use what is learned above
% [U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', 25);

whos imgfeat imgid imglab

%% now compare classification
Xtest = imgfeat';

Rtest = Xtest'*U'*V*Y;
p_all = compute_perf(Rtest(:), 1.*full(imglab(:)), 'store_raw_pr', 2);


[~, n] = size(Xtest);
norm_X = sum(X.^2, 1);  % 1 x n vector
Rtest_k = zeros(size(Rtest));
for i = 1 : n
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

%disp(tag_dim)
%disp([ap_concept, prior_concept] )
[~, ik] = sort( prior_concept, 'descend' );
for i = 1 : ntag
    jj = ik(i);
    fprintf(1, '%0.4f\t %0.4f\t %0.4f\t %s \n', ap_concept(jj, :), prior_concept(jj), tag_dim{jj});
end

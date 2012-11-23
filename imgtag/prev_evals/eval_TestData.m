
NUMV = 150 ;
NUMI = [5000   20000   3000  1000]; %[100   200   300  1000]; 
% NUMI[1:2] # of imgs with 5+ tags, # with <=5 tags ==> training data
% NUMI[3:4]

K = 5; %[3 5 7] ; %9:-2:5;
alph = 100 ; %[1 10 100 1000 10000]; 
max_iter = 10; %25

%% -----------
addpath ../commontools/
dbstop if error

[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/nuswide2';
elseif strcmp(hostn, 'cantabile')
    data_dir = '/home/xlx/data/imgnet-flickr/nuswide2';
else % mac os
    data_dir = '/Users/xlx/Documents/proj/imgnet-flickr/nuswide';
end


%% re-learn the model
%load( fullfile(data_dir, 'explore', 'eval_multilabel_20120411T162020.mat'))

sav_dir = fullfile(data_dir, 'run-data');
tt = datestr(now, 30);
sav_file = fullfile(sav_dir, ['eval_TestDataAll_' tt '.mat']);
cur_diary = fullfile(sav_dir, ['eval_TestDataAll_' tt '.diary']);
diary(cur_diary)

fprintf('logging to %s\n', cur_diary);
fprintf('settings:\n\tNUMV=%d\n', NUMV);
fprintf('\tNUMI= %s\n', num2str(NUMI) );
fprintf('\tK= %s\n', num2str(K) );
fprintf('\talpha= %s\n', num2str(alph));
fprintf('\tmax_iter= %d\n', max_iter);

% set random state
rng(1);

% load up data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'train_label', 'train_idx_map') 

%% setup Y
tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore');

[vs, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
tag_feat = tag_feat(:, iv(1: NUMV));

%% setup X
feat_data_mat = fullfile(data_dir, 'Train_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(train_idx_map, imgid));

imglab = train_label(img_idx, found_wn) ;
tag_dim = tag81(found_wn);

% subselect some data
ntag = full(sum(imglab,2));
nlen = length(ntag);

id1 = rand_idx(find(ntag>=5), NUMI(1)) ;  % these are image ids/row sample
tmplab1 = imglab(id1, :);
ind1 = sample_ind(tmplab1, 2, 2);

id2 = rand_idx(find(ntag<5), NUMI(2)) ;
tmplab2 = imglab(id2, :);
ind2 = sample_ind(tmplab2, 5, 2); % 5 times neg ids

id3 = rand_idx(setdiff((1:nlen)', [id1;id2]), NUMI(3)) ;

id5 = rand_idx(setdiff(find(ntag>=5), [id1;id2]), NUMI(4)) ;

cur_lab = [tmplab1; tmplab2] ;
% convert ids for ind1
for t = 1 : length(ind1)
    [tmpi, tmpj] = ind2sub(size(tmplab1), ind1{t});
    ind1{t} = sub2ind(size(cur_lab), tmpi, tmpj);
end
% shift ids for ind2
for t = 1 : length(ind2)
    [tmpi, tmpj] = ind2sub(size(tmplab2), ind2{t});
    tmpi = tmpi + NUMI(1); % shift nrows down
    ind2{t} = sub2ind(size(cur_lab), tmpi, tmpj);
end
id_train = [ind1{1}; ind2{1}];

R = sparse(zeros(size(cur_lab)) );
R(id_train) = cur_lab(id_train)-.5; % \pm . 5, 0 are unknowns

id_test = [ind1{2}; ind2{2}];
%Rtest = sparse(zeros(size(tmplab1)));
%R(id_test) = cur_lab(id_test)-.5;

X = imgfeat([id1;id2], :)';
Y = log(tag_feat + 1)';

save(sav_file) 

fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));
        
[U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', max_iter, 'solver', 'lbfgs');

%% evaluate
Rt(id_test) = cur_lab(id_test)-.5;  % test targets
Rt = Rt(:);
Re = X'*U'*V*Y;
mse_train = sum((R(id_train)-Re(id_train)).^2)/length(id_train);
mse_test = sum((Rt(id_test)-Re(id_test)).^2)/length(id_test);
fprintf(1, ' test mse %f, training mse %f\n', mse_test, mse_train);

po = compute_perf(Re(id_test), 1.*full(cur_lab(id_test)), 'store_raw_pr', 2);

%% implement KNN baseline

%[ti, tj] = ind2sub(size(X), id_test);
[~, n] = size(X);
norm_X = sum(X.^2, 1);  % 1 x n vector
R_knn = zeros(size(Re));
for i = 1 : n
    dx = -2*X(:, i)'*X + norm_X;
    [~, ix] = sort(dx);
    R_knn(i, :) = sum(R(ix(1:K), :), 1);
end
pk = compute_perf(R_knn(id_test), 1.*full(cur_lab(id_test)), 'store_raw_pr', 2);

fprintf(1, ' test AP with matchbox %0.4f, with knn %0.4f, prior %0.4f\n', po.ap, pk.ap, po.prior);

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

%% setup Y ...
% tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
% load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore'); 
% [vs, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
% tag_feat = tag_feat(:, iv(1: NUMV));


%% setup X
feat_data_mat = fullfile(data_dir, 'Test_feat_objbank.mat');
load(feat_data_mat, 'imgid', 'imgfeat');
img_idx = cell2mat(values(test_idx_map, imgid));

imglab = test_label(img_idx, found_wn) ;
tag_dim = tag81(found_wn);
ntag = length(tag_dim);
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

disp(tag_dim)
disp([ap_concept, prior_concept] )

save(sav_file) 
diary off

% if 0
%     fprintf(1, ' AP on new data w. 5+ tags: matchbox %0.4f, knn %0.4f, prior %0.4f\n', p5.ap, pk5.ap, p5.prior);
%     % evaluate top K precision
%     R5_k = R5_k + 0.001*rand(size(R5_k)); % randomly break ties
%     [tp5, tk5] = deal(ones(n,1));
%     topk = 5;
%     for i = 1 : n
%         jl = find(lab5(i, :)); % groundtruth
%         [~, jn] = sort(R5(i, :),  'descend');
%         [~, jk] = sort(R5_k(i, :), 'descend');
%         tp5(i) = length(intersect(jn(1:topk), jl))/topk;
%         tk5(i) = length(intersect(jk(1:topk), jl))/topk;
%     end
%     fprintf(1, '%s top%d precision on 5+ tags: \n', datestr(now,31));
%     fprintf(1, '  matchbox: %0.4f, [%s]', mean(tp5), num2str(hist(tp5, 0:.2:1)));
%     fprintf(1, '  k-nn    : %0.4f, [%s]', mean(tk5), num2str(hist(tk5, 0:.2:1)));

%     fprintf(1, ' AP on allTest w. 5+ tags: \n\tmatchbox %0.4f, knn %0.4f\n',...
%         p5.ap, p5pca.ap, pk5.ap);

% end
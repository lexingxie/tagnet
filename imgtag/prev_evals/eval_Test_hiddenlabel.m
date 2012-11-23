
NUMV = 150 ;
NUMI = [1000   2000   3000  1000]; %[100   200   300  1000]; 

K = 5; %[3 5 7] ; %9:-2:5;
alph = 100 ; %[1 10 100 1000 10000]; 
max_iter = 2; %25

tag_iv = [11 20];

%% -----------
addpath ../commontools/
dbstop if error

[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/nuswide';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/nuswide';
end

%% re-learn the model
%load( fullfile(data_dir, 'explore', 'eval_multilabel_20120411T162020.mat'))

sav_dir = fullfile(data_dir, 'run-data');
tt = datestr(now, 30);
sav_file = fullfile(sav_dir, ['TestDataAll_' tt '.mat']);
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

% read tag list
tag_seq = textread(fullfile(data_dir, 'tag_seq_63.txt'), '%s');
fprintf('\t evaluating tags:\n');
disp(tag_seq(tag_iv(1):tag_iv(2))')

% load up data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'train_label', 'train_idx_map') 

% setup Y
tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore');

[vs, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
tag_feat = tag_feat(:, iv(1: NUMV));

% setup X
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
[ir, jr] = ind2sub(size(cur_lab), id_train);

id_test = [ind1{2}; ind2{2}];
%Rtest = sparse(zeros(size(tmplab1)));
%R(id_test) = cur_lab(id_test)-.5;

X = imgfeat([id1;id2], :)';
Y = log(tag_feat + 1)';


for i = tag_iv(1) : tag_iv(2)
    cur_tag = tag_seq{i};
    [p, n, e] = fileparts(sav_file);
    cur_sav_file = fullfile(p, [sprintf('%02d_%s_', i, cur_tag), n, e]);
    ii = strmatch(cur_tag, tag_dim, 'exact');

    R = sparse(zeros(size(cur_lab)) );
    jflag = (jr~=ii);
    R(id_train(jflag)) = cur_lab(id_train(jflag))-.5; % \pm . 5, 0 are unknowns

    fprintf('\n\n =========== tag %d: "%s" , %d/%d deleted tags ==============\n', i, cur_tag, sum(jflag==0), length(jflag));
    fprintf(' K=%d, alpha=%f, num-v=%d \n\n', K, alph, size(Y,1));
    
    save(cur_sav_file) 
    
    [U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', max_iter);

    %% load entire test data list and labels
    load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'test_label', 'test_idx_map')
    %... , 'test_label', 'test_idx_map');


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
    clear imgfeat imgid test_idx_map test_label

    Rtest = Xtest'*U'*V*Y;
    p_all = compute_perf(Rtest(:), 1.*full(imglab(:)), 'store_raw_pr', 2);

    % knn
    % [~, n] = size(Xtest);
    % norm_X = sum(X.^2, 1);  % 1 x n vector
    % Rtest_k = zeros(size(Rtest));
    % for i = 1 : n
    %     dx = -2*Xtest(:, i)'*X + norm_X;
    %     [~, ix] = sort(dx);
    %     Rtest_k(i, :) = sum(R(ix(1:K), :), 1);
    % end
    % pktest = compute_perf(Rtest_k(:), 1.*full(imglab(:)), 'store_raw_pr', 2);

    fprintf(1, ' AP on ALL-TEST matchbox %0.4f, prior %0.4f\n', p_all.ap, p_all.prior);

    % compute per=concept AP for both
    ap_concept = zeros(ntag, 2);
    prior_concept = zeros(ntag, 1);
    for i = 1 : ntag
        p = compute_perf(Rtest(:,i), 1.*full(imglab(:,i)), 'store_raw_pr', 2);
        ap_concept(i,1) = p.ap;
        prior_concept(i) = p.prior;
        %pk = compute_perf(Rtest_k(:,i), 1.*full(imglab(:,i)), 'store_raw_pr', 2);
        %ap_concept(i,2) = pk.ap;
    end
    fprintf(1, ' MAP on allTest: \n\tmatchbox %0.4f, knn %0.4f, prior %0.4f\n',...
        mean(ap_concept), mean(prior_concept));

    % implement simple voting/averaging from known tags
    cur_tf = Y(:, ii);
    td = sum( (cur_tf*ones(1, length(tag_dim)) - Y).^2, 1) ; % tag feature distance
    [~, itd] = sort(td) ;
    jj = itd(2:4) ;
    fprintf(1, 'closest features: '); disp(tag_dim (jj));
    rvote = mean(Rtest(:, jj), 2) ;
    pv = compute_perf(rvote, 1.*full(imglab(:,ii)), 'store_raw_pr', 2);

    fprintf(1, ' AP on %s: matchbox %0.4f, voting %0.4f, prior %0.4f\n', ...
        cur_tag, ap_concept(ii, 1), pv.ap, prior_concept(ii) );

    save(cur_sav_file) 
    
end

diary off


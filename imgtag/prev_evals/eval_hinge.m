%% EXP 1 -- tag recommendataion, or multi-tag labeling

NUMV = 150;
NUMI = [100   200   300  1000]; %[200 200 3000 3000];

Klist = 5; %[3 5 7] ; %9:-2:5;
alpha_list = 100 ; %[1 10 100 1000 10000]; 

%% -----------
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/nuswide2';
elseif strcmp(hostn, 'cantabile')
    data_dir = '/home/xlx/data/imgnet-flickr/nuswide2';
else % mac os
    data_dir = '/Users/xlx/Documents/proj/imgnet-flickr/nuswide';
end

sav_dir = fullfile(data_dir, 'run-data');
tt = datestr(now, 30);
sav_file = fullfile(sav_dir, ['eval_mbhinge_' tt '.mat']);
cur_diary = fullfile(sav_dir, ['eval_mbhinge_' tt '.diary']);
diary(cur_diary)

fprintf('logging to %s\n', cur_diary);
fprintf('settings:\n\tNUMV=%d\n', NUMV);
fprintf('\tNUMI= %s\n', num2str(NUMI) );
fprintf('\tK= %s\n', num2str(Klist) );
fprintf('\talpha= %s\n', num2str(alpha_list));


% set random state
tmp = version('-release');
v = str2double(tmp(1:4));
if v >= 2011
    rng(1);
else
    rand('twister', 1);
end

% load up data list and labels
load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag81', 'train_label', 'train_idx_map') 
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
%R(id_train) = cur_lab(id_train)-.5; % \pm . 5, 0 are unknowns
R(id_train) = 2*cur_lab(id_train)-1; % -1 or 1, 0 are unknowns

id_test = [ind1{2}; ind2{2}];
%Rtest = sparse(zeros(size(tmplab1)));
%R(id_test) = cur_lab(id_test)-.5;

X = imgfeat([id1;id2], :)';
Y = log(tag_feat + 1)';

Xnew = imgfeat(id3, :)';
labnew = imglab(id3, :);

X5 = imgfeat(id5, :)';
lab5 = imglab(id5, :);

clear imgfeat train_label train_idx_map
save(sav_file) % cache everything


%% optimize
for K = Klist
    for alph = alpha_list
        fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));
        
        %[U, V] = matchbox_hinge(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', 25);
        [U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', 25);
        
        %% evaluate
        Rt(id_test) = 2*cur_lab(id_test)-1;  % test targets
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
        
        fprintf(1, ' test AP with matchbox_hinge %0.4f, with knn %0.4f, prior %0.4f\n', po.ap, pk.ap, po.prior);
        
        %% test on new data
        Rn = Xnew'*U'*V*Y;
        mse_new = sum((full(labnew(:))-.5-Rn(:)).^2)/length(Rn(:));
        fprintf(1, ' new test mse %f \n', mse_new);
        
        pn = compute_perf(Rn(:), 1.*full(labnew(:)), 'store_raw_pr', 2);
        
        [~, n] = size(Xnew);
        norm_X = sum(X.^2, 1);  % 1 x n vector
        R_nk = zeros(size(Rn));
        for i = 1 : n
            dx = -2*Xnew(:, i)'*X + norm_X;
            [~, ix] = sort(dx);
            R_nk(i, :) = sum(R(ix(1:K), :), 1);
        end
        pkn = compute_perf(R_nk(:), 1.*full(labnew(:)), 'store_raw_pr', 2);
        
        fprintf(1, ' AP on new data: with matchbox_hinge %0.4f, with knn %0.4f, prior %0.4f\n', pn.ap, pkn.ap, pn.prior);
        
        %% test on new, 5+ tag data
        R5 = X5'*U'*V*Y;
        %mse_new = sum((full(labnew(:))-.5-Rn(:)).^2)/length(Rn(:));
        %fprintf(1, ' new test mse %f \n', mse_new);
        
        p5 = compute_perf(R5(:), 1.*full(lab5(:)), 'store_raw_pr', 2);
        
        [~, n] = size(X5);
        %norm_X = sum(X.^2, 1);  % 1 x n vector
        R5_k = zeros(size(R5));
        for i = 1 : n
            dx = -2*X5(:, i)'*X + norm_X;
            [~, ix] = sort(dx);
            R5_k(i, :) = sum(R(ix(1:K), :), 1);
        end
        pk5 = compute_perf(R5_k(:), 1.*full(lab5(:)), 'store_raw_pr', 2);
        
        fprintf(1, ' AP on new data w. 5+ tags: matchbox_hinge %0.4f, knn %0.4f, prior %0.4f\n', p5.ap, pk5.ap, p5.prior);
        % evaluate top K precision
        R5_k = R5_k + 0.001*rand(size(R5_k)); % randomly break ties
        [tp5, tk5] = deal(ones(n,1));
        topk = 5;
        for i = 1 : n
            jl = find(lab5(i, :)); % groundtruth
            [~, jn] = sort(R5(i, :),  'descend');
            [~, jk] = sort(R5_k(i, :), 'descend');
            tp5(i) = length(intersect(jn(1:topk), jl))/topk;
            tk5(i) = length(intersect(jk(1:topk), jl))/topk;
        end
        fprintf(1, '%s top%d precision on 5+ tags: \n', datestr(now,31));
        fprintf(1, '  matchbox_hinge: %0.4f, [%s]\n', mean(tp5), num2str(hist(tp5, 0:.2:1)));
        fprintf(1, '  k-nn    : %0.4f, [%s]\n', mean(tk5), num2str(hist(tk5, 0:.2:1)));
        
        save(sav_file) % save results after optimization (only get to keep the last round though)
    end
end

diary off
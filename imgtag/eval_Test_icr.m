
function eval_Test_icr(Y_mat, cat_or_add, num_data_sample, ntag_collab_f, K)

if nargin < 3
    num_data_sample = inf;
end

if nargin < 4
    ntag_collab_f = 0;
end

if nargin < 5
    K = 5;
end


[~, ystr, ~] = fileparts(Y_mat);

eval_str = ['eval_Test' ystr '_' cat_or_add '_cf' num2str(ntag_collab_f) '_'] ;

exp_envsetup
exp_setparams


% if use_norm
%     Ycache_mat = fullfile(data_dir, 'Y_CN5_prnorm.mat');
% else
%     Ycache_mat = fullfile(data_dir, 'Y_CN5_pr.mat');
% end
Ycache_mat = fullfile(data_dir, Y_mat);
load(Ycache_mat, 'Y', 'Y5p', 'Yadd') %, 'Y5g', 'Yadg');

switch lower(cat_or_add)
    case 'cat'
        Y = [Y(1:NUMV, :); Y5p(1:NUMV, :)] ;
    case 'add'
        % add features
        Y = Yadd(1:NUMV, :);
    case 'none'
        Y = Y5p(1:NUMV, :) ;
    case 'base'
        Y = Y(1:NUMV, :);
    case 'eye'
        Y = eye(size(Y, 2));
    otherwise 
        disp(cat_or_add);
        error(' dunno which feature to take, error!')
        
end

whos

disp(eval_str)

if ntag_collab_f==0 % regular tagging eval
    disp(num_data_sample)
    exp_setupTraining
    
    echo on
    %% learn the model
    fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));
    id_train = find(R(:));
    [U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', max_iter, 'solver', 'lbfgs');
    
    echo off
    
    exp_runTest
else % setup training / test to measure precision@5, with at most ntag_collab_f tags in training
    COLLAB_PREC_D = floor(5 + ntag_collab_f);
    
    exp_setupCollabF
end

clear imgfeat imgid imglab X* R

save(sav_file)
diary off


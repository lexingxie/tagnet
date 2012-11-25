
function eval_TestCN5pr(use_norm, cat_or_add)

if use_norm
    eval_str = 'eval_TestCN5pr_norm_' ;
else
    eval_str = 'eval_TestCN5pr_unrm_' ;
end

eval_str = [eval_str cat_or_add '_'];

exp_envsetup
exp_setparams


num_data_sample = inf;

whos

disp(eval_str)

if use_norm
    Ycache_mat = fullfile(data_dir, 'Y_CN5_prnorm.mat');
else
    Ycache_mat = fullfile(data_dir, 'Y_CN5_pr.mat');
end
load(Ycache_mat, 'Y', 'Y5p', 'Yadd');

switch lower(cat_or_add)
    case 'cat'    
        Y = [Y(1:NUMV, :); Y5p(1:NUMV, :)] ;
    case 'add'
        % add features
        Y = Yadd(1:NUMV, :);
    case 'none'
        Y = Y5p(1:NUMV, :) ;
    otherwise 
        disp(' dunno which feature to take, error!')
        return
end

exp_setupTraining

echo on
%% learn the model
fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));        
id_train = find(R(:));
[U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', max_iter, 'solver', 'lbfgs');

echo off

exp_runTest

clear imgfeat imgid imglab X* R

save(sav_file) 
diary off


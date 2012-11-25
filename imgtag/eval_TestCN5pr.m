
eval_str = 'eval_TestCN5pr_' ;

exp_envsetup
exp_setparams

echo on

num_data_sample = inf;

whos

%Ycache_mat = fullfile(data_dir, 'Y_CN5pr.mat');
Ycache_mat = fullfile(data_dir, 'Y_CN5_prnorm.mat');
load(Ycache_mat, 'Y', 'Y5p');

Y = [Y(1:NUMV, :); Y5p(1:NUMV, :)] ;

exp_setupTraining

%% learn the model
fprintf('\n\n=========== K=%d, alpha=%f, num-v=%d ==============\n\n', K, alph, size(Y,1));        
id_train = find(R(:));
[U, V] = matchbox(R, X, Y, alph, 'indR', id_train, 'k', K, 'max_iter', max_iter, 'solver', 'lbfgs');

exp_runTest

clear imgfeat imgid imglab X* R

save(sav_file) 
diary off
echo off

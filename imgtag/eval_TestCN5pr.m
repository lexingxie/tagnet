
function eval_TestCN5pr(Y_mat, cat_or_add, num_data_sample, K)

[~, ystr, ~] = fileparts(Y_mat);

eval_str = ['eval_Test' ystr '_' cat_or_add '_'] ;

exp_envsetup
exp_setparams

if nargin < 3
    num_data_sample = inf;
end

if nargin < 4
    K = 5;
end

whos

disp(eval_str)
disp(num_data_sample)

% if use_norm
%     Ycache_mat = fullfile(data_dir, 'Y_CN5_prnorm.mat');
% else
%     Ycache_mat = fullfile(data_dir, 'Y_CN5_pr.mat');
% end
Ycache_mat = fullfile(data_dir, Y_mat);
load(Ycache_mat, 'Y', 'Y5p', 'Yadd', 'Y5g', 'Yadg');

switch lower(cat_or_add)
    case 'cat'
        Y = [Y(1:NUMV, :); Y5p(1:NUMV, :)] ;
    case 'gcat'
        Y = [Y(1:NUMV, :); Y5g(1:NUMV, :)] ;
    case 'add'
        % add features
        Y = Yadd(1:NUMV, :);
    case 'gadd'
        % add graph features
        Y = Yadg(1:NUMV, :);
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


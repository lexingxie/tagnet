

function eval_Test_dimY(sort_y)

switch sort_y
    case 'vcnt'
        eval_str = 'eval_Test_Ycnt_' ;
    case 'vscore'
        eval_str = 'eval_Test_Yscore_' ;
    case 'rand'
        eval_str = 'eval_Test_Yrand_' ;
    otherwise
        fprintf(1, ' dunno know how to select Y, quit');
        return;
end

exp_envsetup
exp_setparams

whos

%echo on

num_data_sample = inf;
disp(eval_str)

tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore', 'target_tags');

switch sort_y
    case 'vcnt'
        [~, iv] = sort(vcnt, 'descend'); % take the most frequent
    case 'vscore'
        [~, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
    case 'rand'
        [~, iv] = sort(rand(1,length(vscore)), 'descend'); % random
    otherwise
        fprintf(1, ' dunno know how to select Y, quit');
        return;
end

Y = tag_feat(:, iv(1: NUMV));
Y = log(Y + 1)';

% %Ycache_mat = fullfile(data_dir, 'Y_CN5pr.mat');
% Ycache_mat = fullfile(data_dir, 'Y_CN5_prnorm.mat');
% load(Ycache_mat, 'Y', 'Y5p', 'Yadd');
%
% % cat features
% %Y = [Y(1:NUMV, :); Y5p(1:NUMV, :)] ;
% % add features
% Y = Yadd(1:NUMV, :);

%echo off
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

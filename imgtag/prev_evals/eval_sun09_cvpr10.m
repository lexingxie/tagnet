

ref_dir = '/Users/xlx/proj/ImageNet/sun09';
eval_mat_name = 'sun09_eval.mat';
if exist(fullfile(ref_dir, eval_mat_name), 'file')
    load(fullfile(ref_dir, eval_mat_name));
else
    arg_list = {'presence_score', 'presence_score_c', 'presence_truth', 'names'};
    load(fullfile(ref_dir, 'results_cvpr10.mat'), arg_list{:});
    load(fullfile(ref_dir, 'sun09_train_test.mat'), 'test_flag', 'objname', 'objname_mapped', 'new_objname');
    
    % map object name
    objn_invidx = cell(length(new_objname), 1);
    for i = 1 : length(names)
        ii = strmatch(names{i}, objname, 'exact');
        jj = strmatch(objname_mapped{ii}, new_objname, 'exact');
        objn_invidx{jj} = [objn_invidx{jj}, i];
    end
    
    fprintf(1, ' %d objects excluded\n', sum(cellfun('isempty', objn_invidx)) );
    
    eval_objname = new_objname(~cellfun('isempty', objn_invidx));
    objn_invidx = objn_invidx(~cellfun('isempty', objn_invidx));
    eval_obj_flag = ~cellfun('isempty', objn_invidx) ;
    num_obj = length(eval_objname);
    
    [base_score, model_score] = deal( zeros(sum(test_flag), num_obj) );
    truth_label = false(sum(test_flag), num_obj) ;
    for j = 1 : num_obj
        base_score(:, j) = max(presence_score(objn_invidx{j}, test_flag), [], 1)';
        model_score(:, j) = max(presence_score_c(objn_invidx{j}, test_flag), [], 1)';
        truth_label(:, j) = max(presence_truth(objn_invidx{j}, test_flag), [], 1)';
    end

    save(fullfile(ref_dir, eval_mat_name), 'base_score', 'model_score', 'truth_label', ...
        'eval_objname', 'eval_obj_flag', 'objn_invidx');
end

%% --------------------------------
%% your eval runs from here

num_obj = length(eval_objname);

p_all = compute_perf([base_score(:), model_score(:)], 1.*truth_label(:), 'store_raw_pr', 2);

fprintf(1, 'Overall AP - baseline: %0.4f, model: %0.4f, prior: %0.4f\n', p_all.ap, p_all.prior);


% per-object evaluation
ap_log = zeros(num_obj, 4);
for j = 1 : num_obj
    p = compute_perf([base_score(:, j), model_score(:, j)], 1.*truth_label(:, j), 'store_raw_pr', 2);
    ap_log(j, :) = [p.prior, p.ap, diff(p.ap)];
end

[~, jj] = sort(ap_log(:,1), 'descend'); % sort by prior, descending
bar(ap_log(jj, :) );
axis tight; grid on;
legend({'prior', 'baseline', 'context-model'});
fprintf(1, 'mean-AP over %d objects: prior %0.4f, baseline %0.4f, model %0.4f, diff %0.4f\n', num_obj, mean(ap_log, 1));


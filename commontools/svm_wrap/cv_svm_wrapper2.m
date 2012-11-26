function [mod_final, info] = cv_svm_wrapper2(label, obs, varargin)
% L-fold cross-validation for hyperparameter grid search
% algorithms include: SVM, multi-kernel machines
% label: Nx1 +1/-1
% data: {DxN} (K)  feature vector, 
%       or kernel matrix (NxN) in case of self-supplied kernel
% 
% 2006-08-14, xlx
% 2011-10-01 add liblinear for linear sparse SVM

[sv_method, num_folds, Cpen, gam, do_normalize, search_metric, ... %learning_algo, , kernel_func, ...
    kernel_type, ~, ap_depth, ~, cv_sample_id, neg_pos_ratio, max_num_pos, max_num_neg, ... 
    ~, feature_args, testlabel, testobs] = process_options(varargin, ...
    'sv_method', 'svm', 'num_folds', 3, 'Cpen', 10.^(0:3), 'gam', [10 50 -.5 -3 -4 -5], 'do_normalize', true, ...
    'search_metric', 'ap', ...    %'learning_algo', 'mkm', 'kernel_func', @compute_kernel, ...
    'kernel_type', 2, 'svm_est_prob', 0, 'ap_depth', -1, 'tol', 1e-8, ...
    'cv_sample_id', {}, 'neg_pos_ratio', 0, 'max_num_pos', 1e4, 'max_num_neg', 2e4, ... 
    'feature_func', @seq_avg, 'feature_args', {}, 'testlabel', [], 'testobs', []);

% number of data points, labels must be + 1 / -1
N = length(label) ;
if iscell(obs)
    assert(length(obs)==N, 'data size do not agree!');
else
    assert(size(obs,1)==N, 'data size do not agree!');
end
if strcmpi(sv_method, 'svm')
    assert(all(unique(label)' == [0 1]), 'input label to cv_wrapper() must be 0/1') ;
    cur_label = 1.*(label==1) - 1.*(label==0) ; % convert to +1/-1
else % SVR
    cur_label = label;
end


if kernel_type == 0 || kernel_type < 0
    % linear SVM: kt=0 for libsvm solver, kt=-1 for liblinear solver
    gam = 0;
end
% kernel_type = -3 -- epsilon-SVR
if kernel_type==-3 && ~any(strcmp(search_metric, {'tau', 'mse', 'corr'}))
    fprintf(1, 'KERNEL_TYPE=%d, regression, set SEARCH_METRIC to "TAU"\n', kernel_type);
    search_metric = 'tau';
    
end

num_C = length(Cpen);
num_gam = length(gam);
%if any(gam) 
    ipar1 = ones(num_gam,1)*(1:num_C) ;
    ipar2 = (1:num_gam)' * ones(1,num_C);
    ipar = [ipar1(:), ipar2(:)]';
    clear ipar1 ipar2
%end

num_param = size(ipar, 2);


if isempty(cv_sample_id)
    if strcmpi(sv_method, 'svm')
        cv_sample_id = montecarlo_cv_sample(1:N, ones(1, num_folds)/num_folds, label);
    else
        cv_sample_id = montecarlo_cv_sample(1:N, ones(1, num_folds)/num_folds);
    end
else
    assert(size(cv_sample_id,2)==num_folds, 'supplied cross-validation id does not agree!');
end

if size(ipar, 2)>1    
    
    %for nt = 1 : NT  % number of ground truth
    
    perf_log = zeros(num_folds, size(ipar, 2));
    
    for nf = 1 : num_folds
        % get the n-th fold, including the +tive and -tive ids
        if size(cv_sample_id,1)>1
            test_id = union(cv_sample_id{:,nf}); % 
        else
            test_id = cv_sample_id{:,nf};
        end
        train_id = setdiff(1:N, test_id);
        
        traind.label = cur_label(train_id);
        testd.label = cur_label(test_id);
        
        % resample negative training data if needed
        if strcmp(sv_method, 'svm') && neg_pos_ratio > 0
            ineg = find(traind.label==-1);
            ipos = find(traind.label== 1);
            if length(ineg)>neg_pos_ratio*length(ipos)
                ii = randperm(length(ineg));
                ineg = ineg(ii);
                jneg = ineg(1: neg_pos_ratio*length(ipos));
            else
                jneg = ineg ;
            end
            fprintf(1, 'resampled negatives from %d to %d, %0.2f x #pos=%d\n', ...
                length(ineg), length(jneg), neg_pos_ratio, length(ipos));
            if length(ipos) > max_num_pos
                tmp = randperm(length(ipos));
                ipos = ipos(tmp(1:max_num_pos));
                fprintf(1, 'capped positive examples at %d\n', max_num_pos);
            end
            if length(jneg) > max_num_neg
                tmp = randperm(length(jneg));
                jneg = ineg(tmp(1:max_num_neg));
                fprintf(1, 'capped negative examples at %d\n', max_num_neg);
            end
            train_id = train_id([ipos; jneg]);
            traind.label = cur_label(train_id);
        end
        
        traind.mat = obs(train_id, :) ;
        testd.mat  = obs(test_id, :);
        
        feat_model = {};
        %[traind.mat, feat_model] = feature_func(obs(train_id, :), [], feature_args);
        %testd.mat = feature_func(obs(test_id, :), feat_model, feature_args);
        %         end
        model.feature_model = feat_model ;
        model.feature_args = feature_args ;
        if ~isempty(feat_model), fprintf(1, '\n %s, finished generating features for fold #%d \n', datestr(now, 31), nf); end
        
        % normalize data
        if do_normalize
            [traind.mat, model.normpar] = svm_box_data(traind.mat, []) ;
            testd.mat = svm_box_data(testd.mat, model.normpar);
        else
            model.normpar.min = zeros(1, size(traind.mat, 2));
            model.normpar.max = ones (1, size(traind.mat, 2));
            %npar.min = zeros(1,d);
            %npar.max = ones (1,d);
        end
        
        ee = cputime;
        fprintf(1, ' cross-validation fold %d/%d \n', nf, num_folds);
        all_dist = tril(xminusy_square(traind.mat')) ;
        all_dist = all_dist(abs(all_dist)>eps);
        %% perform parameter search
        for nc = 1 : num_param
            % train-test- report stats
            %ee = cputime;
            %fprintf(1, ' cross-val [%d,%d]/[%d,%d] \n', nf, nc, num_folds, num_param);
            cur_gamma = gam(ipar(2, nc)) ;
            cur_c = Cpen(ipar(1, nc)) ;
            
            %switch learning_algo,   case 'svm'
            if cur_gamma > 0
                %cur_gamma = cur_gamma/sigma_sq;
                % use Alex Smola heuristic for kernel width: 
                % median and certain percentile of the data distance         
                cur_gamma = 1/prctile(all_dist, cur_gamma) ;
                % 1/d because of libSVM implementation
                % K(x1,x2)=-exp(-gamma|x1-x2|^2)
                svm_options = sprintf('-g %f -t 2',  cur_gamma);
            elseif cur_gamma<0 && round(cur_gamma)==cur_gamma
                % polynomial kernel
                svm_options = sprintf('-t 1 -d %d', -cur_gamma);
            elseif kernel_type < 0 && round(kernel_type)==kernel_type
                svm_options = sprintf('-s %d', -kernel_type); 
            else % linear kernel, libSVM solver
                svm_options = '-t 0';
            end
            if strcmpi(sv_method, 'svr') && kernel_type>=0 % SVR
                svm_options = sprintf('%s -s 3', svm_options);
            end
            svm_options = sprintf('%s -c %f -q', svm_options, cur_c);
            
            if kernel_type < 0 && round(kernel_type)==kernel_type % liblinear solver
                model.svm = train(traind.label, 1.*(traind.mat), svm_options);                
                [pred_label, ~, pred_val] = predict(testd.label, 1.*testd.mat, model.svm);
            else % libsvm solver for both SVC and SVR
                model.svm = svmtrain(traind.label, 1.*(traind.mat), svm_options);                
                [pred_label, ~, pred_val] = svmpredict(testd.label, 1.*testd.mat, model.svm);
            end
            pred_val = abs(pred_val).*pred_label;
            
            if strcmpi(sv_method, 'svm')
                test_perf = compute_perf(pred_val, testd.label==1, 'ap_depth', ap_depth, 'store_raw_pr', 2);
            else
                test_perf = compute_perf_regression(pred_val, testd.label);
            end
            
            par_log(nc).gam = cur_gamma;
            par_log(nc).cpen = cur_c;
            
            %otherwise
            %        fprintf(1, 'unknown learning algorithm %s \n', learning_algo);
            %end % switch algo
            
            %fprintf(1, '  ap %0.3f, prec@%d %0.3f, epr %0.3f (radom %0.4f)\n', ...
            %    test_perf.ap, test_perf.pr_dp, test_perf.p_at_d, test_perf.epr, test_perf.prior);
            
            eval(sprintf('perf_log(nf, nc) = test_perf.%s ;', search_metric));
        end % num_par
        
        disp(reshape(perf_log(nf, :), num_gam, []));
        fprintf(1, '\b %s, cv-traintest time %0.4fs \n',  datestr(now, 31), cputime - ee);
    end % num_fold
    
else
    disp('no need to search');
end
clear model

%% find the best config with stat. confidence
mp = mean(perf_log, 1);
if length(mp)>1
    [~, ii] = sort(mp);   ii = ii([end end-1]);
    [h, hp] = ttest(perf_log(:, ii(1)), perf_log(:, ii(2)));
    if max(mp)-min(mp)<0.01   % not much difference among runs
        ii = num_gam*num_C ;  % use linear kernel with the largest C
    elseif h == 1 || all(perf_log(:, ii(1))-perf_log(:, ii(2)) > eps)
        ii = ii(1);
    else
        ii = ii(1); %ii(1 + round(rand));
    end
else % use linear kernel
    ii = num_gam*num_C-1 ; h = 0;
end

cstar = par_log(ii).cpen;
gstar = par_log(ii).gam;

if size(ipar, 2)>1, disp(perf_log); end
fprintf(1, '\nperf metric (mean %s) : ',  upper(search_metric));
fprintf(1, '%0.3f ',  sort(mp, 'descend'));
fprintf(1, '\nmean perf, gamma, Cpen : \n');
disp(reshape(mp, num_gam, [])) ;    disp([]);
disp(reshape([par_log.gam], num_gam, [])) ;    disp([]);
disp(reshape([par_log.cpen], num_gam, [])) ;

mod_final.search = struct('stat_conf', hp, 'cpen', Cpen, 'gam', gam, 'mp', mp);
mod_final.cpen = cstar;
mod_final.gam = gstar;

%% ------- train final model with all data --------
if isnumeric(cstar) && isnumeric(gstar)
    fprintf(1, '\n%s SVM best param: C=%f g=%f (p=%0.4f)\n\n',  datestr(now,30), cstar, gstar, hp);
end

%[mod_final.norm, fstat, traind] = prep_featdata_norm(data, cv_label, do_normalize, kt_list, 1:N);
%clear data

%% generate features 
traind.mat = obs;
feat_model = {};
%[traind.mat, feat_model] = feature_func(obs, [], feature_args);
traind.label = cur_label; %(train_id);
mod_final.feature_model = feat_model;
mod_final.feature_args = feature_args ;

if do_normalize
    % normalize data
    [traind.mat, mod_final.normpar] = svm_box_data(traind.mat, []) ;
else
    
    model.normpar.min = zeros(1, size(traind.mat, 2));
    model.normpar.max = ones (1, size(traind.mat, 2));
    %npar.min = zeros(1,d);
    %npar.max = ones (1,d);
end
fprintf(1, ' %s, finished generating features for all training data\n', datestr(now, 31));
   
%% learn SVM
%switch learning_algo
%    case 'svm'       

svm_options = sprintf('-q -c %f', cstar);
if gstar > 0
    %all_dist = tril(xminusy_square(traind.mat)) ;
    %all_dist = all_dist(abs(all_dist)>eps);
    %sigma_sq = mean(mean( xminusy_square(traind.mat) )); %mean(traind.mat(:).^2);
    %cur_gamma = cur_gamma/sigma_sq;
    mod_final.method = 'svm'; %learning_algo;

    svm_options = sprintf('-t 2 -g %f %s', gstar, svm_options);
elseif gstar<0 && round(gstar)==gstar
    svm_options = sprintf('-t 1 -d %d %s', -gstar, svm_options);
elseif kernel_type < 0 && round(kernel_type)==kernel_type
    mod_final.method = 'liblinear';
    svm_options = sprintf('-s %d', -kernel_type);
else % linear kernel, libsvm solver
    mod_final.method = 'svm-lin'; %learning_algo;
    svm_options = sprintf('-t 0 %s', svm_options);
end
if strcmpi(sv_method, 'svr') && kernel_type>=0 % SVR
    mod_final.method = 'svr';
    svm_options = sprintf('%s -s 3', svm_options);
end

mod_final.options = svm_options;
if kernel_type < 0 % liblinear solver
    mod_final.svm = train(traind.label, 1.*(traind.mat), svm_options);
    [pred_label, ~, pred_val] = predict(testd.label, 1.*testd.mat, mod_final.svm);
else % libsvm solver
    mod_final.svm = svmtrain(traind.label, 1.*traind.mat, svm_options);
    [pred_label, ~, pred_val] = svmpredict(traind.label, 1.*traind.mat, mod_final.svm);
end

pred_val = abs(pred_val).*pred_label;

fprintf(1, ' final svm options: %s \n', svm_options);
if strcmpi(sv_method, 'svm')
    train_perf = compute_perf(pred_val, traind.label==1, 'ap_depth', ap_depth, 'store_raw_pr', 2);
    %test_perf = compute_perf(pred_val, testd.label==1, 'ap_depth', ap_depth, 'store_raw_pr', 2);
    fprintf(1, ' train_set ap %0.3f, auc %0.3f, prior %0.3f \n', ...
        train_perf.ap, train_perf.auc, train_perf.prior);
    mod_final.train_AP = train_perf.ap;
else
    train_perf = compute_perf_regression(pred_val, traind.label);
    fprintf(1, ' train_set corr %0.3f, tau %0.3f, mse %0.3f \n', ...
        train_perf.corr, train_perf.tau, train_perf.mse);
    %mod_final.train_AP = train_perf.ap;
    eval(sprintf('mod_final.train_%s = train_perf.%s ;', search_metric, search_metric));
end



if ~isempty(testlabel) && ~isempty(testobs)
    testmat = svm_box_data(testobs, mod_final.normpar);
    [pred_label, ~, pred_val] = svmpredict(1.*testlabel, 1.*testmat, mod_final.svm);
    pred_val = abs(pred_val).*pred_label;
    
    if strcmpi(sv_method, 'svm')
        test_perf = compute_perf(pred_val, testlabel==1, 'ap_depth', ap_depth, 'store_raw_pr', 2);
        fprintf(1, '%s TEST set AP %0.3f, AUC %0.3f, prior guess %0.3f \n\n', ...
            datestr(now,30), test_perf.ap, test_perf.auc, test_perf.prior);
    else %svr
        test_perf = compute_perf_regression(pred_val, traind.label);
        fprintf(1, '%s TEST set corr %0.3f, tau %0.3f, mse %0.3f \n', ...
            datestr(now,30), test_perf.corr, test_perf.tau, test_perf.mse);
    end

end

fprintf(1, '\n');

%% pack some extra returns

try
    info.perf_log = perf_log ;
    info.search_metric = search_metric ;
    info.param_log = par_log ;
catch
    disp('extra info not furnished');
end

if exist('test_perf', 'var')
    info.test_perf = test_perf;
    info.test_pred = pred_val;
end
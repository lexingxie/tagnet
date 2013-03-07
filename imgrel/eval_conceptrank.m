%% evaluate relation prediction with a predictor (here bigram value baseline, mutual info baseline)
function rperf = eval_conceptrank(target, rankmat, exclude_mat, varargin)

[tau, nrounds, prec_depth, verbose, eval_mode] = process_options(varargin, ...
    'tau', 0, 'nrounds', 1, 'prec_depth', 100, 'verbose', 1, 'eval_mode', 'micro');

n = size(target, 1);
assert(size(rankmat,1)==n, 'target and input size disagree!');
jnz = find(sum(rankmat, 1) ~= 0);  % indexes of covered concepts

switch eval_mode
    case 'macro'
        include_flag = true(n) & ~exclude_mat>0;
        rperf.idx = jnz ;
        rperf.ap = jnz ;
        rperf.f1 = jnz ;
        rperf.p_at_d = jnz ;
        rperf.prior = jnz ;
        rperf.npos = jnz ;
        for k = 1: length(jnz)
            j = jnz(k);
            jscore = rankmat(j, include_flag(j,:) ) ;
            jlabel = (target(j, include_flag(j,:) ) >0)*1. ;
            perf = compute_perf(jscore, jlabel, 'store_raw_pr', 2, 'precision_depth', prec_depth);
            
            rperf.prior(k) = perf.prior ;
            rperf.npos(k) = perf.npos ;
            rperf.ap(k) = perf.ap ;
            rperf.f1(k) = perf.f1 ;
            rperf.p_at_d(k) = perf.p_at_d ;
            
        end
        
    case 'micro'
        include_mat = triu(true(n), 1) & ~exclude_mat>0;
        include_idx = find(include_mat) ;
        
        target_mat = triu(target, 1)>0 & include_mat>0 ;
        tlabel = 1.* target_mat(include_idx) ;
        
        % compute max. possible recall
        nzmat = false(size(rankmat));
        nzmat(jnz, jnz) = true;
        nzmat = nzmat & (target_mat>0) ;
        cpos = sum(nzmat(:));
        cnz = length(jnz);
        
        num_inc = length(include_idx);
        %num_target = nnz(target_mat) ;
        [ap, f1, auc, p_at_d] = deal(zeros(1,nrounds) );
        for j = 1 : nrounds
            rscore = rankmat(include_idx) + rand(num_inc,1)*tau ;   % break ties randomly
            perf = compute_perf(rscore, tlabel, 'store_raw_pr', 2, 'precision_depth', prec_depth);
            
            ap(j) = perf.ap;
            f1(j) = max(perf.f1) ;
            auc(j) = perf.auc ;
            p_at_d(j) = perf.p_at_d;
        end
        rperf.max_recall = cpos/sum(tlabel) ;
        rperf.r_prior = 2*cpos/(cnz*(cnz-1)) ;
        
        rperf.npos = perf.npos;
        rperf.prior = perf.prior;
        rperf.ap = [mean(ap), std(ap)];
        rperf.auc = [mean(auc), std(auc)];
        rperf.f1 = [mean(f1), std(f1)];
        rperf.p_at_d = [mean(p_at_d), std(p_at_d)];
        
        [rs, ir] = sort(rscore, 'descend') ;
        % i, j, score, truth
        itmp = ir(1:prec_depth);
        [j1, j2] = ind2sub([n n], include_idx(itmp)) ;
        rperf.top_returns = [ j1, j2, rs(1:prec_depth), tlabel(itmp) ];
        
        if verbose
            fprintf(1, ' prior: %0.4f, npos: %d\n', rperf.prior, perf.npos);
            fprintf(1, ' AP: %0.4f +- %0.4f\n', rperf.ap );
            fprintf(1, ' AUC: %0.4f +- %0.4f\n', rperf.auc );
            fprintf(1, ' F1: %0.4f +- %0.4f\n', rperf.f1 );
            fprintf(1, ' P@%d: %0.4f +- %0.4f\n', prec_depth, rperf.p_at_d );
        end
        
    otherwise
        fprintf(1, 'unknown eval_mode= %s, QUIT\n\n', eval_mode);
end
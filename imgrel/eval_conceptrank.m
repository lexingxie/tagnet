%% evaluate relation prediction with a predictor (here bigram value baseline, mutual info baseline)
function rperf = eval_conceptrank(target, rankmat, exclude_mat, varargin) 

[tau, nrounds, prec_depth, verbose] = process_options(varargin, ...
    'tau', .1, 'nrounds', 10, 'prec_depth', 100, 'verbose', 1);

n = size(target, 1);
assert(size(rankmat,1)==n, 'target and input size disagree!');
include_mat = triu(ones(n), 1) & ~exclude_mat>0;
include_idx = find(include_mat) ;

target_mat = triu(target, 1)>0 & include_mat>0 ;
tlabel = 1.* target_mat(include_idx) ; 

num_inc = length(include_idx);
num_target = nnz(target_mat) ;
[ap, f1, p_at_d] = deal(zeros(1,nrounds) );
for j = 1 : nrounds
    rscore = rankmat(include_idx) + rand(num_inc,1)*tau ; % break ties randomly
    perf = compute_perf(rscore, tlabel, 'store_raw_pr', 2, 'precision_depth', prec_depth);
    
    ap(j) = perf.ap;
    f1(j) = max(perf.f1) ;
    p_at_d(j) = perf.p_at_d;
end
rperf.prior = perf.prior;
rperf.ap = [mean(ap), std(ap)];
rperf.f1 = [mean(f1), std(f1)];
rperf.p_at_d = [mean(p_at_d), std(p_at_d)];

if verbose
    fprintf(1, ' prior: %0.4f, npos: %d\n', rperf.prior, perf.npos);
    fprintf(1, ' AP: %0.4f +- %0.4f\n', rperf.ap );
    fprintf(1, ' F1: %0.4f +- %0.4f\n', rperf.f1 );
    fprintf(1, ' P@%d: %0.4f +- %0.4f\n', prec_depth, rperf.p_at_d );
end
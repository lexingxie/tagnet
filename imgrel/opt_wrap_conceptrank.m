function gw = opt_wrap_conceptrank(val_lb, init_g, obsf, varargin)
% optimization wrapper for concept-rank

% alph is the "restart probablity" in pagerank [default .5]
%     affect the random walk length and eigen value separability (when
%     closer to 1). some value .3 ~ .85 should be safe choice
%   used as "hatw = alph*w + (1-alph)* ones(n,1) * v' ; "
% alphR is the trade-off between L2 regularization and loss function [.1]
%     this needs to be tuned using training data

[alph, alphR, gradobj, maxiter, solver, bfgs_ttits] = process_options(varargin, ...
    'alph', .5, 'alphR', .1, 'GradObj', 'off', 'maxiter', 50, 'solver', 'lbfgs', 'bfgs_ttits', 100);


idx = find(init_g) ; %sub2ind([n n], [a(1,:) a(2,:)], [a(2,:) a(1,:)]); 
cdx = find(val_lb);
n = size(init_g, 1);

options = optimset('Algorithm', 'active-set','GradObj', gradobj, 'MaxIter', maxiter, 'display', 'off');%, 'DerivativeCheck', 'on');
opts.printEvery = 1;
opts.maxits = maxiter;
opts.maxTotalIts = bfgs_ttits ;

[A, b, Aeq, beq] = deal([]);
lb = full(val_lb(idx)) ;
ub = ones(size(lb)) ;

vargs = {'alphR', alphR, 'alph', alph, 'gradobj', gradobj, 'relaxed', true};
fr = @(g) conceptrank_wrap (g, idx, n, obsf, vargs{:}) ;

fprintf(1, '%s start optimization: #vars %d, #initial relations %d \n', datestr(now, 31), length(idx), length(cdx));
% optimize G
if strcmpi(solver, 'lbfgs')
    opts.x0 = init_g(idx(:));
    [g1, fval, info] = lbfgsb(fr, lb, ub, opts);
else
    [g1, ~] = fmincon(fr, val_lb(idx), A, b, Aeq, beq, lb, [], [], options) ;
end

fprintf(1, '%s Done. \n', datestr(now, 31));
% initial f
fprintf(1, ' initial f\n');
[~] = conceptrank_wrap (init_g(idx), idx, n, obsf, 'VERBOSE', 'final', vargs{:}) ;
% final f
fprintf(1, ' final f:\n');
[~] = conceptrank_wrap (g1, idx, n, obsf, 'VERBOSE', 'final', vargs{:}) ;

% put back to square form
gw = sparse(zeros(n)); % reconstruct G matrix, symmetric
gw(idx) = g1;

function wn_concept_rank(varargin)

[wnid, exp_home, out_d, win_d, topK] = process_options(varargin, ...
    'wnid', 'n01680983', 'exp_home','/Users/xlx/Documents/proj/imgnet-flickr', ...
    'out_subdir', 'optm-out', 'wnet_in_dir', 'wnet-out', 'topK', 15) ;

out_mat = fullfile(exp_home, out_d, [wnid '.mat']);
load(fullfile(exp_home, win_d, [wnid '.mat']), 'csubnet', 'mi', 'freq', 'tag_list');

[cnet, cignore, obsf, obsm] = convert_wn_input(csubnet, freq, mi) ;

% produce some entries for evaluation

g = opt_conceptrank(cnet, cignore, obsf) ;


% print some info for sense-making
print_top_pairs(tril(g + g'), tag_list, topK, c(1:2,:), 'ConceptRank') ;

print_top_pairs(tril(obsm), tag_list, topK, c(1:2,:), 'mutual info') ;

print_top_pairs(tril(obsf), tag_list, topK, c(1:2,:), 'cooc freq') ;

print_top_pairs(tril(cnet), tag_list, topK, c(1:2,:), 'conceptnet') ;


save(out_mat);



% -------------------
function gw = opt_conceptrank(val_lb, cignore, init_g, varargin)
%alph = .5;
%alphR = .1;
[alph, alphR, gradobj] = process_options(varargin, 'alph', .5, 'alphR', .1, 'GradObj', 'off');


idx = find(init_g) ; %sub2ind([n n], [a(1,:) a(2,:)], [a(2,:) a(1,:)]); 

options = optimset('GradObj', gradobj, 'MaxIter', 50, 'display', 'iter', 'DerivativeCheck', 'on');

[A, b, Aeq, beq] = deal([]);
lb = full(val_lb(idx)) ;

vargs = {'alphR', alphR, 'alph', alph, 'ignore_mask', cignore, 'gradobj', false, 'relaxed', true};
fr = @(g) conceptrank_wrap (g, idx, n, obsf, vargs{:}) ;


fprintf(1, '%s start optimization: #vars %d, #initial relations %d \n', datestr(now, 31), length(idx), size(c, 2));
% optimize G
[g1, ~] = fmincon(fr, val_lb(idx), A, b, Aeq, beq, lb, [], [], options) ;


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


% ----- function to convert wn input to matrixes -----
function [cnet, cignore, obsf, obsm] = convert_wn_input(csubnet, freq, mi) %, tag_list)

c = double(reshape(csubnet, 3, []));
c(1:2, :) = c(1:2, :) + 1;

a = reshape(double(freq), 3, []);
b = reshape(mi, 3, []);
a(1:2, :) = a(1:2, :) + 1; % zero-based indexing in python --> 1-based indexing here
b(1:2, :) = b(1:2, :) + 1;

n = double(max([a(1,:), a(2,:), c(1,:), c(2,:)]) ) ;

cnet = sparse(c(1,:), c(2,:), c(3,:), n, n);
cnet = 1.*(cnet>0) ;
cnet = cnet + cnet' ;
cignore = c(1:2, c(3,:)<0); % index of the ignored positions

obsm = sparse(b(1,:), b(2,:), b(3,:), n, n);
obsm = obsm + obsm' ;
[obsm, ~] = normalise(obsm, 2);

obsf = sparse(a(1,:), a(2,:), a(3,:), n, n);
obsf = obsf + obsf' ;
[obsf, ~] = normalise(obsf, 2);
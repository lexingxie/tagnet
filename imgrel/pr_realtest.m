

%load /Users/xlx/Documents/proj/imgnet-flickr/wnet-out/n04524313.mat 
wnid = 'n01680983'; % sagebrush_lizard.n
exp_home = '/Users/xlx/Documents/proj/imgnet-flickr';
out_mat = fullfile(exp_home, 'optm-out', [wnid '.mat']);
load(fullfile(exp_home, 'wnet-out', [wnid '.mat']), 'csubnet', 'mi', 'freq', 'tag_list');

topK = 15;

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

alph = .5;
alphR = .1;


%fr = @(g) conceptrank_obj (g, obsf, 'alphR', alphR, 'alph', alph, 'ignore_mask', cignore, 'gradobj', false) ;
idx = sub2ind([n n], [a(1,:) a(2,:)], [a(2,:) a(1,:)]); 

options = optimset('GradObj', 'off', 'MaxIter', 50, 'display', 'iter', 'DerivativeCheck', 'on');

[A, b, Aeq, beq] = deal([]);
lb = full(cnet(idx)) ;

vargs = {'alphR', alphR, 'alph', alph, 'ignore_mask', cignore, 'gradobj', false, 'relaxed', true};
fr = @(g) conceptrank_wrap (g, idx, n, obsf, vargs{:}) ;
%[e, err] = gradest(fr, G0(:));

fprintf(1, '%s start optimization: #vars %d, #initial relations %d \n', datestr(now, 31), length(idx), size(c, 2));
% optimize G
[g1, fval] = fmincon(fr, cnet(idx), A, b, Aeq, beq, lb, [], [], options) ;


fprintf(1, '%s Done. \n', datestr(now, 31));
% initial f
fprintf(1, ' initial f\n');
fi = conceptrank_wrap (obsf(idx), idx, n, obsf, 'VERBOSE', 'final', vargs{:}) ;
% final f
fprintf(1, ' final f:\n');
fe = conceptrank_wrap (g1, idx, n, obsf, 'VERBOSE', 'final', vargs{:}) ;

% print some info for sense-making

gw = zeros(n); % reconstruct G matrix, symmetric
gw(idx) = g1;

print_top_pairs(tril(gw + gw'), tag_list, topK, c(1:2,:), 'ConceptRank') ;

print_top_pairs(tril(obsm), tag_list, topK, c(1:2,:), 'mutual info') ;

print_top_pairs(tril(obsf), tag_list, topK, c(1:2,:), 'cooc freq') ;

print_top_pairs(tril(cnet), tag_list, topK, c(1:2,:), 'conceptnet') ;

save(out_mat) ;

% [~, ig] = sort(gg(:), 'descend');
% [gi, gj] = ind2sub([n,n], ig(1:topK));
% fprintf(1, ' Top-ranked %d concept pairs: \n', topK);
% for i = 1 : topK
%     ik = gi(i); jk = gj(i);
%     if any(c(1,:)==ik & c(2,:)==jk)
%         statustr = '(konwn)' ;
%     else
%         statustr = '(-)' ;
%     end
%     fprintf(1, '\t %0.4f\t %s %s %s\n', gg(ig(i)), tag_list(gi(i),:), tag_list(gj(i), :), statustr) ;
% end



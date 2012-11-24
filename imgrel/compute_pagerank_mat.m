
function [Gout] = compute_pagerank_mat(Gin, alph)

nw = size(Gin, 1);

Gout = zeros(nw);
e = ones(nw,1);

for c = 1 : nw
    v = 1. * ((1:nw)'== c) ;
    hatw = alph*Gin + (1-alph)* e * v' ;
    [p, ~] = eigs(hatw', 1);
    Gout(c, :) = p / sum(p) ;
    if mod(c, 100)==0
        fprintf(1, '%s done computing %d / %d pagerank\n', datestr(now, 31), c, nw);
    end
end

function [Gout] = compute_pagerank_mat(Gin, alph, renorm_method)
% compute personalized pagerank

% normalize the input, not the output
if nargin<3 || strcmpi(renorm_method, 'logistic')
    renorm_func = inline( '.5*(1-exp(-x))./(1+exp(-x))' ); % logistic function
elseif  strcmpi(renorm_method, 'log')
    renorm_func = inline('log10(1+x)'); %logscaling
else
    % do nothing
end

Gin = renorm_func(Gin); 

nw = size(Gin, 1);

Gout = zeros(nw);
e = ones(nw,1);

parfor c = 1 : nw
    v = 1. * ((1:nw)'== c) ;
    hatw = alph*Gin + (1-alph)* e * v' ;
    [p, ~] = eigs(hatw', 1);
    Gout(c, :) = p / sum(p) ;
    if mod(c, 100)==0
        fprintf(1, '%s done computing %4d / %4d pagerank\n', datestr(now, 31), c, nw);
    end
end



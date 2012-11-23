function [J, pJ] = conceptrank_wrap (g, idx, n, Zbar, varargin)
% wrapper over conceptrank_obj() to reduced dimension of vars (to len(idx) )

%[loss_func, alphR, alph, ignore_mask, gradobj, relaxed, DEBUG] = process_options(varargin, ...
%    'loss_func', @l2norm, 'alphR', 1, 'alph', .5, 'ignore_mask', 1, ...
%    'gradobj', true, 'relaxed', false, 'DEBUG', false) ;

%idx = sub2ind([n n], ig, jg); 
%
% this wrapper can technically be merged into conceptrank_obj, but to avoid
% more substaintial change i'm keep it for now -- xlx

numg = length(idx) ;
if numg < n*n/4
    G = sparse(n, n);
else
    G = zeros(n);
end

% convert g back as an n x n matrix 
G(idx) = g ;

[J, pJ] = conceptrank_obj(G, Zbar, 'idx_g', idx, varargin{:}); 
%'alphR', alphR, 'alph', alph, 'gradobj', false) ;

if ~isempty(pJ)
    pJ = pJ(idx);
end
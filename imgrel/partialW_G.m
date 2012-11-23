
% ------
function [ww, pwg, dnm] = partialW_G(G)
% derivative from normalization recipe w_{ij}=\frac{g_{ij}}{\sum_k g_{ik}}
% if \sum_k g_{ik} = 0 then w_{ik} = 1/n, 

n = size(G, 1);
sg = sum(G, 2) ;
iz = sg < eps;

ww = G ./ ( (sg + ~sg) * ones(1, n) ) ; % normalize G
ww(iz, :) = 1/n;
sg(iz) = 1; %n*eps ;

pwg = - ww ./ (sg(:)*ones(1,n)) ;
dnm = 1./sg(:);
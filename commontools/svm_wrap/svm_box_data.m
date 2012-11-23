function [nx, npar] = svm_box_data(x, par)
% box-normalize input x (N-by-d) to [0, 1] in each dimension

if islogical(x) || (isstruct(par) && all(par.min==0) && all(par.max==1))
    [~,d] = size(x);
    
    npar.min = zeros(1,d);
    npar.max = ones (1,d);
    nx = x;
else
    if nargin<2 || isempty(par)
        npar.min = min(x, [], 1);
        npar.max = max(x, [], 1);
    else
        [npar.min, npar.max] = deal(par.min, par.max);
    end
    
    nrange = npar.max - npar.min ;
    nrange = nrange + (nrange<eps)*1 ;
    
    n = size(x, 1);
    nx = (x - ones(n,1)*npar.min) ./ (ones(n,1)*nrange) ;  %[0,1]
    %nx = 2* (nx - .5) ;
end
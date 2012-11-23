function [J, pJ] = conceptnorm_obj (ww, Zbar, varargin)
% this function computes the objective function:
%    \min_G~J_R = \frac{\alpha_R}{2} ||G||^2 + \sum_{u,v} h(z_{uv} - \bar z_{uv})
% and its partial derivative w.r.t. g_ij
%    \alpha_R g_{ij} +  
%    \sum_{u,v} \fracP{h(\delta_{uv})}{\delta_{uv}} \fracP{z_{uv}}{w_{ij}} \fracP{w_{ij}}{g_{ij}}
% 

[loss_func, alphR, alph, relaxed, DEBUG] = process_options(varargin, ...
    'loss_func', @l2norm, 'alphR', 1, 'alph', .5, 'relaxed', false, 'DEBUG', false) ;

if size(ww, 2) == 1
    % input is of the form g = G(:), convert back to square
    n = sqrt(length(ww)); 
    ww = reshape(ww, n, n);
    flatten_output = true ;
else
    n = size(ww, 1);
    assert(n == size(ww, 2), 'matrix G is not square!');
    flatten_output = false ;
end
if 1 %~relaxed
    if ~all( ww(:)>= -eps )
        ww
    end
    assert(all( ww(:)>= -eps ), 'matrix G is not >= 0 ');
    assert(all(abs(sum(ww, 2) - 1)< n*eps('single') ), 'matrix is not stochastic ');
end

% first term of both J and partial-J
J = 0; %.5*alphR*norm(ww);
pJ = zeros(n) ; %alphR * G ;

%tmp = sum(G, 2); 
%ww = G ./ ( (tmp + ~tmp) * ones(1, n) ) ; % normalize G
%[ww, pwg] = partialW_G(G) ; % normalize G, and compute partial_W / partial G

for ii = 1 : n
    [z, pz] = compute_ppr(ww(:), alph, ii, 'eigen', [], 50, 0) ;
    %if DEBUG
        % test if the derivatives are correct
    %    fr = @(ww) compute_ppr(ww, alph, ii, 'power', [], 50, 0) ;
        %[e, err] = gradest(fr, ww(:));
    %end
    
    delta = z - Zbar(ii, :)' ;   
    [los, plos] = loss_func(delta) ;
    
    J = J + los ;
    %pJ = pJ + pwg .* reshape( plos'*pz, n, n ) ;
    pJ = pJ + reshape( plos'*pz, n, n ) ;
end

if flatten_output
    pJ = pJ(:) ;
end

% ------
function [x, px] = l2norm(delta)

x = delta(:);
x = x'*x ;

px = 2*delta(:);







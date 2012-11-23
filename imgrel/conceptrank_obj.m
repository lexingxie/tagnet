function [J, pJ1] = conceptrank_obj (G, Zbar, varargin)
% this function computes the objective function:
%    \min_G~J_R = \frac{\alpha_R}{2} ||G||^2 + \sum_{u,v} h(z_{uv} - \bar z_{uv})
% and its partial derivative w.r.t. g_ij
%    \alpha_R g_{ij} +  
%    \sum_{u,v} \fracP{h(\delta_{uv})}{\delta_{uv}} \sum_k \fracP{z_{uv}}{w_{ik}} \fracP{w_{ik}}{g_{ij}}
% 

[loss_func, alphR, alph, idx_g, gradobj, relaxed, VERBOSE] = process_options(varargin, ...
    'loss_func', @l2norm, 'alphR', 1, 'alph', .5, 'idx_g', [], ...
    'gradobj', true, 'relaxed', false, 'VERBOSE', false) ;

if size(G, 2) == 1
    % input is of the form g = G(:), convert back to square
    n = sqrt(length(G)); 
    G = reshape(G, n, n);
    flatten_output = true ;
else
    n = size(G, 1);
    assert(n == size(G, 2), 'matrix G is not square!');
    flatten_output = false ;
end
if ~relaxed
    assert(all( G(:)>= 0 ), 'matrix G is not >= 0 ');
end
    
if isempty(idx_g)
    idx_g = 1:n*n ;
end
numg = length(idx_g) ;


% first term of both J and partial-J
if issparse(G)
    tmp = full(G(G>0));
    J = .5 * alphR * (tmp' * tmp) ;
else
    J = .5*alphR* sqrt(G(:)'*G(:)) ; % norm(G);
end
if gradobj
    pJ = alphR * G ;
else
    pJ = [] ;
end

J0 = J ; % first part of the objective fun

%tmp = sum(G, 2); 
%ww = G ./ ( (tmp + ~tmp) * ones(1, n) ) ; % normalize G
[ww, pwg, dnm] = partialW_G(G) ; % normalize G, and compute partial_W / partial G

for ii = 1 : n
    if gradobj
        [los, p_grad] = compute_ppr(ww, alph, ii, Zbar(ii,:), 1, loss_func, 'eigen', idx_g, 100, 0) ;
        
        sub_grad = zeros(n);        
        sub_grad(idx_g) = p_grad ;
    else
        [los] = compute_ppr(ww, alph, ii, Zbar(ii, :), 1, loss_func, 'eigen', idx_g, 100, 0) ;
    end
    
    J = J + los ;
    
    if gradobj
        pJ = pJ + diag(dnm)*sub_grad + repmat(diag(sub_grad*pwg'),1,n);
%         for rid = 1:n
% %             chn_grad = dnm(rid)*eye(n) + repmat(pwg(rid,:)',1,n);
%             pJ(rid,:) = pJ(rid,:) + dnm(rid)*sub_grad(rid,:) + sub_grad(rid,:)*pwg(rid,:)';
%         end
    end
end


if gradobj 
    if numg < n*n
        % zero-ing out the gradient for fixed elements 
        % is this necessary?, or has the computation above took care of this already?
        pJ1 = zeros(n); 
        pJ1(idx_g) = pJ(idx_g) ;
    else
        pJ1 = pJ ;
    end    
    
    if flatten_output
        pJ1 = pJ1(:) ;
    end
else
    pJ1 = [];
end

if strcmp(VERBOSE, 'final')
    fprintf(1, ' fval=%f, .5*alphR*|G|=%f, sum_ij L(G) = %f\n', J, J0, J-J0);
end

% ------
function [x, px] = l2norm(delta)

x = delta(:);
x = x'*x ;

px = 2*delta(:);



% if DEBUG
%     test if the derivatives are correct
%     fr = @(ww) compute_ppr(ww, alph, ii, 'power', [], 100, 0) ;
%     [e, err] = gradest(fr, ww(:));
% end




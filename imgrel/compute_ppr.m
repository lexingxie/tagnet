function [los, p_grad] = compute_ppr(w, alph, c, zbar, ignore_self, loss_func, ...
                                        method, indw, numiter, VERBOSE)
% input: normalized matrix W
%        teleportation prob. \alph
%        restart col id c (scalar)
%        assume restart prob \nu_u = e_u
%        zbar is a row vector of observed values zbar = Zbar(c, :) 
% output: loss betweeen personalized pagerank (PPR) vector z and
%           observations zbar
%         C*C^2 partial derivative \fracP{z}{w_ij}

if min(size(w))==1
    n = sqrt(length(w));
    w = reshape(w, n, n);
    relaxed = true;
else
    relaxed = false;
end

if nargin < 2
    alph = .5;
end

if nargin < 3
    c = 1;
end

if nargin < 5
    ignore_self = true;
end
   
if nargin < 6
    method = 'eigs';  % use power iterations or the built-in eigs func
end
n = size(w, 1);
assert(n == size(w, 2), 'matrix W is not square!');

if ~relaxed
    assert(all( abs( sum(w, 2)-1) < n*eps('single') ), 'matrix W is not Markov!');
end

if nargin < 7 || isempty(indw)
    indw = 1:length(w(:)) ;
    return_square_mat = true;
else
    return_square_mat = false;
end
numw = length(indw);

if nargin < 8
    numiter = 5;
end

if nargin < 9
    VERBOSE = 2;
end


% if max(size(ignore_mask)) == 1
%     ignore_self = ignore_mask ; 
%     ignore_idx = false;
% else % a list of ignored indexes
%     ignore_self = true;
%     ignore_idx = true;
% end

% nc = length(c);   % which row to compute
%p = ((1:n)== c)';   % rand(n, 1);

% $\hat W = \alpha W + (1-\alpha) e \nu^T $
v = 1. * ((1:n)'== c) ;
hatw = alph*w + (1-alph)* ones(n,1) * v' ;

switch lower(method)
    case 'eigen'
        [p, ~] = eigs(hatw', 1);
        p = p / sum(p) ;
    case 'power'
        diff = 1e10; 
        iter = 0;
        p_prev = ones(n,1)/n;   
        while diff > eps('single') && iter<numiter
            p = hatw' * p_prev ;
            
            d = p_prev - p ;
            diff = sqrt(d'*d) / n ;
            p_prev = p;     
            iter = iter + 1;
            
            if VERBOSE > 1
                fprintf(1, ' %s PR-iter#%d, diff-p=%0.4e\n', datestr(now, 31), iter, diff);
                if VERBOSE > 2
                    disp(p')
                end
            end
        end
        if VERBOSE
            fprintf(1, ' %s finished #%d power iterations, diff-p=%0.4e\n', datestr(now, 31), iter, diff);
        end
        
    otherwise
        fprintf(1, ' unknown method %s, quit.\n', upper(method));
end


%delta = p - Zbar(c, :)' ;   
delta = p - zbar(:) ;
if ignore_self
    delta(c) = 0; 
end
% if ignore_idx && ~isempty(ignore_mask)
%     igcols = ignore_mask(ignore_mask(:,1)==c, 2);
%     delta(igcols) = 0 ;
% end
[los, plos] = loss_func(delta) ;
        
if nargout < 2
    p_grad = [] ;
else
   
    pp = zeros(1, numw);
    
    [ii, jj] = ind2sub([n, n], indw) ;
    %A1 = [eye(n) - alph*w; ones(1,n) ];
    A = eye(n) - hatw ;
    %% compute group inverse by QR
    [Q_A,R_A] = qr(A);
    U_A = R_A(1:end-1,1:end-1);
    %     check_qr_up = norm(U_A*ones(n-1,1)+R_A(1:end-1,end))
    %     check_qr_dn = norm(R_A(end,:))
     
    e = ones(n,1) ;
    dp = (eye(n)-e*p');
    ginv = U_A\Q_A(:,1:end-1)';
    A_gi = dp*[ginv;zeros(1,n)]*dp;     
    
    A_plos = A_gi*plos;
    pp = alph * A_plos(jj(:)).*p(ii(:));
%     for kk = 1:ceil(length(indw)/n)
%         seg_idx = 1+(kk-1)*n:min(kk*n,length(indw));
%         pp(seg_idx) = alph* (A_gi(jj(seg_idx),:) * plos ).*p(ii(seg_idx));
%     end
    
    if return_square_mat
        p_grad = reshape( pp, n, n );
    else
        p_grad = pp ;        
    end
    
    if VERBOSE > 2
        disp(pp)
    end
end

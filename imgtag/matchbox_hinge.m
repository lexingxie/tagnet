
function [U, V, predR] = matchbox_hinge(R, X, Y, alpha, varargin)

%% optimize R ~ X'*U'*V*Y

[k, indR, initU, initV, szR, fmin_iter, max_iter, ...
    gradobj, DerivativeCheck, epsilon] = process_options(varargin, ...
    'k', 5, 'indR', [], 'initU', [], 'initV', [], 'szR', [], ...
    'fmin_iter', 2, 'max_iter', 5, 'gradobj', 'on', 'DerivativeCheck', 'off', 'epsilon', 1e-5);

%% objective function
% min. \sum_n s_hinge(y * x' UV Z) + .5*\alpha (|U| + |V|)

tmp = version('-release');
v = str2double(tmp(1:4));
if v >= 2011
    rng(1);
else
    rand('twister', 1);
end

if isempty(szR)
    if min(size(R))>1
        szR = size(R);
    else
        error('do not know the size of R!');
    end
end
%[n, m] = size(R);
[p, n] = size(X);
[q, m] = size(Y);
assert(n==szR(1), 'size of X does not agree with R');
assert(m==szR(2), 'size of Y does not agree with R');

if isempty(indR)
    [indR(:,1), indR(:, 2)] = ind2sub(szR, 1 : length(R(:)));
elseif size(indR,2)==1
    [indR(:,1), indR(:, 2)] = ind2sub(szR, indR);
end
if min(size(R))>1
    Rr = full(R(sub2ind(szR, indR(:,1), indR(:, 2)))) ;
else
    Rr = R;
end % pre-process R, only keep the part we need
num_vr = length(Rr);

% pre-process X and Y, keep the parts we need
jx = unique(indR(:, 1));
jy = unique(indR(:, 2));
indRr = indR ;
if length(jx)<n
    rn = length(jx);
    jx_map = [(1:rn)', jx];
    Xr = X(:, jx);
    % map indR
    for i = 1: num_vr
        indRr(i,1) = jx_map(jx_map(:,2)==indR(i,1), 1);
    end
    fprintf('%s shrinking X from %d x %d to %d\n', datestr(now, 31), p, n, rn);
else
    Xr = X;
    jx_map = [1:n; 1:n]';
end
if length(jy)<m
    rm = length(jy);
    jy_map = [(1:rm)', jy];
    Yr = Y(:, jy);
    for i = 1: num_vr
        indRr(i,2) = jy_map(jy_map(:,2)==indR(i,2), 1);
    end
    fprintf('%s shrinking Y from %d x %d to %d\n', datestr(now, 31), q, m, rm);
else
    Yr = Y;
    jy_map = [1:m; 1:m]';
end


%% need better initialization ?
if isempty(initU)
    initU = randn(k, p);
end
if isempty(initV)
    initV = randn(k, q);
end
s = mean(mean(abs(Rr))) / mean(mean(abs(Xr'*initU'*initV*Yr ))); % intial scale
initU = initU*sqrt(s);
initV = initV*sqrt(s);
fprintf('%s scaling initU and initV by %f\n', datestr(now, 31), s);

%%

vec_u = initU(:);
vec_v = initV(:);

tmp = compute_obj(vec_u, vec_v, Rr, Xr, Yr, indRr, alpha, 'U'); 
prev_val = tmp - .5*alpha*(vec_u'*vec_u + vec_v'*vec_v); % this is cheaper
delta = prev_val + 1;
fprintf('%s initial loss= sum_n s_hinge(y * x" U" V Z) =%f, on %d values\n', datestr(now, 31), prev_val, num_vr);

iter = 0;
options = optimset('GradObj', gradobj, 'MaxIter', fmin_iter, ...
    'DerivativeCheck', DerivativeCheck, 'display', 'off') ; %, 'DerivativeCheck', 'on'); 
%options = optimset('GradObj','off', 'MaxIter', 2, 'display', 'off');
while iter < max_iter && abs(delta)>epsilon
    
    fu = @(vu) compute_obj(vu, vec_v, Rr, Xr, Yr, indRr, alpha, 'U') ;
    cur_u = vec_u;
    [vec_u, valU,exitflag,output,grad] = fminunc(fu, cur_u, options);
    
    
    fv = @(vv) compute_obj(vec_u, vv, Rr, Xr, Yr, indRr, alpha, 'V') ;
    cur_v = vec_v;
    [vec_v, valV,exitflag,output,grad] = fminunc(fv, cur_v, options);
    
    %err = compute_err(vec_u, vec_v, Rr, Xr, Yr, indRr) ; 
    err = valV - .5*alpha*(vec_u'*vec_u + vec_v'*vec_v); % this is cheaper
    
    iter = iter + 1;
    delta = valV-prev_val;
    fprintf('%s iter %d, valU=%f, valV=%f, err=%f, delta=%f\n', datestr(now,31), iter, valU, valV, err, delta); %valV-prev_val)
    prev_val = valV;
    %disp(reshape(vec_u, [], size(Xr,1)))
    %disp(reshape(vec_v, [], size(Yr,1)))
end

U = reshape(vec_u, [], size(X,1));
V = reshape(vec_v, [], size(Y,1));
%disp(R)
%disp(X'*U'*V*Y)

if nargout > 2
    predR = X'*U'*V*Y; 
end


%% main err function
%function r_loss(U, V, R, X, Y, indR)

function [obj, grado] = compute_obj(vec_u, vec_v, Rr, X, Y, indR, alpha, target)
%pred = X'*U'*V*Y;
%min. (R(indR) - pred(indR))^2 - alpha*(U**2 + V**2)
num_vr = size(indR, 1); % number of valid R to fit
[p, n] = size(X);
[q, m] = size(Y);

total_loss = 0;
U = reshape(vec_u, [], p);
V = reshape(vec_v, [], q);
k = size(U,1);

assert( length(Rr)==num_vr, 'number of element in Rr must agree with indR');
 
if target=='U'
    grado = alpha*U ;
else
    grado = alpha*V ;
end

% pre-compute u_x and v_y
u_x = U*X; % k x n -- now every X(:, i) is used 
v_y = V*Y; % k x m


% sliced version of x and y, could be expensive if indR large
%xx = X(:, indR(:, 1));
%yy = Y(:, indR(:, 2)); 

%Rr is expected to be {-1 1}
for i = 1 : num_vr
    %cure = Rr(i) - xx(:, i)'*U'*V*yy(:, i) ; % par-for version
    %cure = Rr(i) - X(:, indR(i, 1))'*U'*V*Y(:, indR(i,2)) ; % original version
    [hl, ghl] = smooth_hinge(Rr(i) * u_x(:, indR(i, 1))'*v_y(:, indR(i,2)) ) ; % cached version
    total_loss = total_loss + hl;
    
    % compute gradient
    if target=='U'
        %grado = grado - cure*V*Y(:, indR(i,2))*X(:, indR(i,1))' ;
        %grado = grado - cure*V*yy(:, i)*xx(:, i)' ;
        grado = grado + ghl*Rr(i)*v_y(:, indR(i,2))*X(:, indR(i,1))' ;
        
    else % V
        grado = grado + ghl*Rr(i)*u_x(:, indR(i,1))*Y(:, indR(i,2))' ;
        %grado = grado - cure*U*X(:, indR(i,1))*Y(:, indR(i,2))' ;
        %grado = grado - cure*U*xx(:, i)*yy(:, i)' ;
    end
    
end

obj = total_loss + .5*alpha*(vec_u'*vec_u + vec_v'*vec_v) ;

grado = grado(:);


%% smooth hinge loss function
% http://mathoverflow.net/questions/51370/smooth-approximation-of-the-hinge-loss-function
function [h, gh] = smooth_hinge(x)

nx = length(x);

if x <= 0 
    h = .5 - x ;
    gh = -1*ones(nx, 1);
elseif x < 1
    h = .5*(1 - x).*(1 - x);
    gh = x - 1;
else % x > 1
    [h, gh] = deal(zeros(nx, 1));
    %gh = 0;
end


%% note: pre-compute trick didn't (quite) work
% surprising (matlab caching results?)
% function  total_err = compute_err(vec_u, vec_v, Rr, X, Y, indR)
% num_vr = size(indR, 1); % number of valid R to fit, Rr is a vector
% total_err = 0;
% [p, n] = size(X);
% [q, m] = size(Y);
% U = reshape(vec_u, [], p);
% V = reshape(vec_v, [], q);
% k = size(U,1);
% 
% assert( length(Rr)==num_vr, 'number of element in Rr must agree with indR');
% 
% % pre-compute u_x and v_y
% u_x = zeros(k, n); % k x n -- now assume every X(:, i) is used 
% v_y = zeros(k, m); % k x m
% parfor i = 1 : n
%     u_x(:, i) = U*X(:, i);
% end
% parfor i = 1 : m
%     v_y(:, i) = V*Y(:, i);
% end
% 
% for i = 1 : num_vr
%     cure = Rr(i) - u_x(:, indR(i, 1))'*v_y(:, indR(i,2)) ; 
%     total_err = total_err + cure^2;
% end
% 
% total_err = .5*total_err;





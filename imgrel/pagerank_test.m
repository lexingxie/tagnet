
addpath ../commontools

if 1
    G = [1     1     1 ; ...
        1     0     1 ; ...
        1     1     0 ];
    
    G0 = [1     1     0 ; ...
        1     0     1 ; ...
        0     1     0 ];
else
    G = [1 0; 0 1];
    G0 = [0.5 0; 0 1];
    G1 = [.5 .5; 0 1];
end


alph = .5;
alphR = 0;
n = size(G, 1);

Zbar = zeros(size(G));
[ww, pwg] = partialW_G(G) ; % normalize G, and compute partial_W / partial G
tmpj = eye(n);
for ii = 1 : n
    Zbar(ii, :) = compute_pr(ww, alph, ii, 'eigen', [], 500, 1) ;
    
    %for jj = 1 : n
    %    fr = @(ww) (tmpj(jj, :) * compute_ppr (ww(:), alph, ii, 'eigen', [], 50, 0) ) ;
    %    [pze(jj, :), pzr, stp] = gradest(fr, ww(:));
    %end
end

% [J, pJ] = conceptrank_obj (G, Zbar, 'alphR', alphR, 'alph', alph, 'VERBOSE', 'final') ;
% [J0, pJ0] = conceptrank_obj (G0, Zbar, 'alphR', alphR, 'alph', alph, 'VERBOSE', 'final') ;

% now setup the optimization problem
if 1
    opts.x0 = G0(:) + rand(size(G0(:))) ;    
    lb = G0(:);
    %ub = ones(length(G0(:)), 1) ;
    [A, b, Aeq, beq] = deal([]);
    % ----------------------------
    options = optimset('Algorithm', 'interior-point', 'GradObj', 'on', 'MaxIter', 50, 'display', 'notify', 'DerivativeCheck', 'on');
    
    fr = @(g) conceptrank_obj (g, Zbar, 'alphR', alphR, 'alph', alph) ;
    
    % optimize G
    [g1, fval] = fmincon(fr, G0(:), A, b, Aeq, beq, lb, [], [], options) ;
    
    % look at the new value of the objective function
    [~, ~] = conceptrank_obj (g1, Zbar, 'alphR', alphR, 'alph', alph, 'VERBOSE', 'final') ;
    
    % use LBFGS-B
    %[g, f2, info] = lbfgsb( f, lb, ub, opts ) ;
else
     
    lb = zeros(size(G0(:)));
    %'Algorithm', 'active-set',
    % ------  just fit the squre err on Z w. normalized W ----
    options = optimset('Algorithm', 'active-set', 'GradObj', 'on', 'MaxIter', 50, 'display', 'off', 'DerivativeCheck', 'on');
    
    [A, b] = deal([]);
    Aeq = zeros(n, n*n) ;
    Aeq(1, 1:n:end) = ones(1, n);
    for i = 2 : n
        Aeq(i, :) = circshift(Aeq(i-1, :), [0 1]) ;
    end
    beq = ones(n, 1);
    G0_n = normalise(G0,2);
    fn = @(g) conceptnorm_obj (g, Zbar) ;
    [g1, fval] = fmincon(fn, G0_n(:), A, b, Aeq, beq, lb, [], [], options) ;
end





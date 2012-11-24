

[m, n, p, q, k] = deal(4, 5, 3, 3, 2);

rand('twister', 1);

X = randn(p, n);
Y = randn(q, m);
U = randn(k, p);
V = randn(k, q);
Rval = X'*U'*V*Y ;
R = sign(Rval) ;

U0 = randn(k, p);
V0 = randn(k, q);

[U1, V1, predR1] = matchbox(R, X, Y, 1, 'initU', U0, 'initV', V0) ;

disp('try hinge loss ');

[U2, V2, predR2] = matchbox_hinge(R, X, Y, 1, 'initU', U0, 'initV', V0, 'gradobj', 'on', 'DerivativeCheck', 'on') ;

[U3, V3, predR3] = matchbox_hinge(R, X, Y, 1, 'initU', U0, 'initV', V0, 'solver', 'lbgfs') ;

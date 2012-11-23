
n = 300;
alph = .5 ;
w = rand(n) ; 
v = 1. * ((1:n)'== 2) ; 

K = 5000;

tic; 
for i = 1 : K
    hatw = alph*w + (1-alph)* ones(n,1) * v' ;
end
toc ; 
clear hatw

tic; 
for i = 1 : K
    hatw = alph*w + (1-alph)* v(:, ones(1,n))' ;
end
toc ; 


% slow
% tic;
% for i = 1 : K
%     hatw = alph*w + (1-alph)* repmat(v, 1, n)' ;
% end
% toc ;

% slow too
% tic; 
% for k = 1 : K
%     hatw = alph*w ;
%     for i = 1 : n
%         hatw(i, :) = hatw(i, :) + (1-alph)* v' ;
%     end
% end
% toc ; 

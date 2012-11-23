function u = xminusy_square(x, y)

if nargin<2
    y = x;
    same = true;
else
    same = false;
end

szx = size(x); % D -by- N
szy = size(y);

assert(szx(1)==szy(1), 'data dimension must agree!');

if 0 %islogical(x) && islogical(y) && szx(1)<min(szx(2),szy(2))
    u = zeros(szx(2),szy(2));
    if same
        for i = 1: szx(2)
            for j = 1 : i-1
                u(i,j) = sum(xor(x(:,i),y(:,j))) ;
            end            
        end
        u = u + u' ;
    else        
        for i = 1: szx(2)
            for j = 1 : szy(2)
                u(i,j) = sum(xor(x(:,i),y(:,j))) ;
            end
        end
    end
            
else    
    x2 = sum(x.^2, 1);
    y2 = sum(y.^2, 1);
    
    u = 1.*x' * y ;
    u = ones(szy(2),1)*x2(:)' + y2(:)*ones(1,szx(2)) - 2*u ;
    % u = x2(ones(szy(2),1),:)' + y2(ones(szx(2),1),:) - 2*u ;
end
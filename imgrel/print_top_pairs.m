function print_top_pairs(G, tag_list, topK, known_subi, new_subi, rnk_critera, reverse)

if nargin < 7
    reverse = false;
end

gg = G; %tril(G + G');
if reverse
    [~, ig] = sort(gg(:));
else
    [~, ig] = sort(gg(:), 'descend');
end
n = size(G, 1);
[gi, gj] = ind2sub([n,n], ig(1:topK));

fprintf(1, ' Top %d concept pairs by %s: \n', topK, rnk_critera);
%[c(:,1), c(:,2)] = ind2sub([n n], find(known_subi) );
%[d(:,1), d(:,2)] = ind2sub([n n], find(new_subi) );

Ks = known_subi + known_subi';
Ns = new_subi + new_subi'; 

for i = 1 : topK
    score = full( gg(ig(i)) );
    if ~reverse && score <= 0, 
        break; 
    elseif reverse && score <= 0
        continue;
    end
    
    ik = gi(i); jk = gj(i);
    if Ks(ik, jk) > eps('single')  %any(c(1,:)==ik & c(2,:)==jk) || any(c(1,:)==jk & c(2,:)==ik)
        statustr = '(konwn)' ;
    elseif Ns(ik, jk) > eps('single') %~isempty(d) & ( any(d(1,:)==ik & d(2,:)==jk) || any(d(1,:)==jk & d(2,:)==ik) )
        statustr = '(new-hit)' ;
    else
        statustr = '(-)' ;
    end
    %fprintf(1, '\t %0.4f\t %s %s %s\n', score, tag_list(gi(i),:), tag_list(gj(i), :), statustr) ;
    fprintf(1, '\t %0.4f\t %s, %s %s\n', score, tag_list{gi(i)}, tag_list{gj(i)}, statustr) ;
end

disp('')
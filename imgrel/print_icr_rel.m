function print_icr_rel(G, knownG, newG, jdx, tag_list, topK, print_known)
% print the top new pairs (correct/not) for a given index

if nargin < 7
    print_known = false;
end

nw = size(G, 1);
if isempty(jdx)
    jdx = randi(nw);
end
    
for j = 1 : length(jdx)
    idx = jdx(j);
    % assume symmertic input, descending
    [~, ig] = sort(G(idx, :), 'descend');
    
    cnt = 0;
    i = 0;
    status_str = repmat('a', 1, topK);
    fprintf(1, '%s\t', tag_list{idx});
    while cnt < topK && i < nw
        i = i + 1;
        ii = ig(i);
        if ii==idx, continue; end
        if knownG(idx, ii) > eps('single')
            %status_str = '(o)' ;
            ss = 'o';
            if print_known
                print_flag = 1;
            else
                print_flag = 0;                
            end
        elseif newG(idx, ii) > eps('single')
            %status_str = '(+)' ;
            ss = '+';
            print_flag = 1;
        else
            %status_str = '(-)' ;
            ss = '-';
            print_flag = 1;
        end
        if print_flag
            fprintf(1, '%s\t', tag_list{ii});
            cnt = cnt + 1;
            status_str(cnt) = ss;
        end
    end
    fprintf(1, '\n\t%s\n', status_str);

end
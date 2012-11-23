function print_graph_data(G, Ginfo, row_label, col_label)

ng = length(G);
if iscell(row_label{1}) && length(row_label)==length(G)
    rl_all = row_label; % each graph slice has its own row label
else 
    rl_all = '';
end

for i = 1 : ng
    [ig, jg, s] = find(G{i});
    fprintf(1, 'graph #%d, %d edges\n', i, length(s));
    disp(Ginfo(i, :));
    if ~isempty(rl_all)
        row_label = rl_all{i};
    end
    flag_printed = false(length(ig), 1);
    for k = 1 : length(ig)
        if ~flag_printed(k)
            fprintf(1, ' (%s, %s)\n ', row_label{ig(k)}, col_label{jg(k)});
            flag_printed(k) = true;
            % check if symmertic edge exist
            kk = (ig==jg(k) & jg==ig(k));            
            if any(kk) && ~flag_printed(kk)
                fprintf(1, ' (%s, %s)\n ', row_label{ig(kk)}, col_label{jg(kk)});
                flag_printed(kk) = true;
            end
        end
    end
    %all_label = [row_label(ig); col_label(jg)];
    %fprintf(1, ' (%s, %s)\n ', all_label{:});
    disp('');
end
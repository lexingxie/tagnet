
%% clean + convert conceptnet tuples to sparse matrix

cn4 = textread('/Users/xlx/Documents/proj/imgnet-flickr/db2/cn4_filter_uniq.txt', '%s');
cn5 = textread('/Users/xlx/Documents/proj/imgnet-flickr/db2/cn5_filter_uniq.txt', '%s');

cn4_core = intersect(cn4, cn5);
cn4_diff = setdiff(cn4, cn5);
cn5_diff = setdiff(cn5, cn4);

rel_list = sort(textread('/Users/xlx/Documents/proj/imgnet-flickr/db2/cn4_relation_list.txt', '%s', 'commentstyle', 'shell'));
nr = size(rel_list);

rel_idmap = containers.Map(rel_list, num2cell(1:nr) ); %1:length(rel_list));
word_idmap = containers.Map('INIT', 0);
cur_new_idx = 1;

% first build word id map
for i = 1 : length(cn5)
    [r, w1, w2] = strread(cn5{i}, '%s%s%s', 'delimiter', ',');
    r = r{1}; w1 = w1{1}; w2 = w2{1};
    if ~isKey(word_idmap, w1)
        word_idmap(w1) = cur_new_idx;
        cur_new_idx = cur_new_idx + 1;
    end
    if ~isKey(word_idmap, w2)
        word_idmap(w2) = cur_new_idx;
        cur_new_idx = cur_new_idx + 1;
    end
    j1 = word_idmap(w1);
    j2 = word_idmap(w2);
    jr = rel_idmap(r);
    
end

nw = length(word_idmap);

fprintf(1, 'read %d words \n', nw)

[G5a{1:nr}] = deal(sparse(nw, nw));
for i = 1 : length(cn5)
    [r, w1, w2] = strread(cn5{i}, '%s%s%s', 'delimiter', ',');
    r = r{1}; w1 = w1{1}; w2 = w2{1};
    j1 = word_idmap(w1);
    j2 = word_idmap(w2);
    jr = rel_idmap(r);
    G5a{jr}(j1,j2) = G5a{jr}(j1,j2) + 1;
end
fprintf(1, 'CN5-all : '), disp(cellfun(@nnz, G5a));

[G4c{1:nr}] = deal(sparse(nw, nw));
for i = 1 : length(cn4_core)
    [r, w1, w2] = strread(cn4_core{i}, '%s%s%s', 'delimiter', ',');
    r = r{1}; w1 = w1{1}; w2 = w2{1};
    j1 = word_idmap(w1);
    j2 = word_idmap(w2);
    jr = rel_idmap(r);
    G4c{jr}(j1,j2) = G4c{jr}(j1,j2) + 1;
end
fprintf(1, 'CN4-core: '), disp(cellfun(@nnz, G4c));

[G5d{1:nr}] = deal(sparse(nw, nw));
for i = 1 : length(cn5_diff)
    [r, w1, w2] = strread(cn5_diff{i}, '%s%s%s', 'delimiter', ',');
    r = r{1}; w1 = w1{1}; w2 = w2{1};
    j1 = word_idmap(w1);
    j2 = word_idmap(w2);
    jr = rel_idmap(r);
    G5d{jr}(j1,j2) = G5d{jr}(j1,j2) + 1;
end

fprintf(1, 'CN5-diff: '), disp(cellfun(@nnz, G5d));

save('/Users/xlx/Documents/proj/imgnet-flickr/db2/CN_graph.mat', 'G*', '*idmap');


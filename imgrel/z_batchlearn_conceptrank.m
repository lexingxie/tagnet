

exp_home = '/Users/xlx/Documents/proj/imgnet-flickr/conceptrank-exp';

synset_list = textread(fullfile(exp_home, 'ilsvrc-intersect.txt'), '%s');

for i = 1 : 2 %length(synset_list)
    in_mat_name = ['ilsvrc873/' synset_list{i} '.mat'] ;
    out_mat_name = ['ilsvrc873-out/' synset_list{i} '.mat'] ;
    
    [GW, tag_list, cn4, cn_new, cn5] = ...
        learn_conceptrank('in_file', in_mat_name, 'solver', 'fmin');
    
    save(fullfile(exp_home, out_mat_name), 'GW', 'tag_list', 'cn4', 'cn_new', 'cn5');
end


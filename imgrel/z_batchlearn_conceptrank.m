

%exp_home = '/Users/xlx/Documents/proj/imgnet-flickr/conceptrank-exp';
exp_home = '/home/xlx/data/imgnet-flickr';
cptr = 'conceptrank-exp';

synset_list = textread(fullfile(exp_home, cptr, 'ilsvrc-intersect.txt'), '%s');

for i = 1 : length(synset_list)
    in_mat_name = ['ilsvrc873/' synset_list{i} '.mat'] ;
    if ~exist(fullfile(exp_home, cptr, in_mat_name), 'file')
	continue;
    end
    out_mat_name = ['ilsvrc873-out/' synset_list{i} '.mat'] ;
    
    try
      [GW, tag_list, cn4, cn_new, cn5] = ...
          learn_conceptrank('in_file', in_mat_name, 'exp_home', exp_home, 'solver', 'lbfgs');
    
       save(fullfile(exp_home, out_mat_name), 'GW', 'tag_list', 'cn4', 'cn_new', 'cn5');
    catch
       fprintf(1, ' error analyzing %s\n!', in_mat_name);
    end
end


addpath ../imgtag
exp_envsetup

alph = .5 ;

cn_graph_mat = fullfile(data_dir, '../db2', 'CN_graph.mat');
load(cn_graph_mat, 'G5a', 'word_idmap' );
%{

load('/Users/xlx/Documents/proj/imgnet-flickr/db2/CN_graph.mat')
whos
  Name               Size              Bytes  Class             Attributes

  G4c                1x14            1701088  cell                        
  G5a                1x14            1981568  cell                        
  G5d                1x14            1156608  cell                        
  rel_idmap         14x1                 112  containers.Map              
  word_idmap      7797x1                 112  containers.Map   
%}

nr = length(G5a);
nw = double(word_idmap.Count) ;
G5 = G5a{1} ;
for r = 2 : nr
    G5 = G5 + G5a{r} ;
end

G5p = zeros(nw);
e = ones(nw,1);
for c = 1 : nw
    v = 1. * ((1:nw)'== c) ;
    hatw = alph*G5 + (1-alph)* e * v' ;
    [p, ~] = eigs(hatw', 1);
    G5p(c, :) = p / sum(p) ;
    if mod(c, 100)==0
        fprintf(1, '%s done computing %d / %d pagerank\n', datestr(now, 31), c, nw);
    end
end

pr_graph_mat = fullfile(data_dir, '../db2', 'CN5_pr.mat');
save(pr_graph_mat, 'G5', 'G5p', 'word_idmap', 'alph');

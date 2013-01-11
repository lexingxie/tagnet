
function [GW, tag_list, cn4, cn_new, cn5] = learn_conceptrank(varargin)
% Optional parameters:
% obs_type controls what observation it fits: "bigram", or conceptnet-5
% entry_type controls what the variables are: "bigram-only", or all

[in_file, exp_home, db_subdir, exp_subdir, topK, ...
    alph, gradobj, solver, bfgs_ttits, obs_type, init_type, entry_type] = process_options(varargin, ...
    'in_file', 'n01680983', 'exp_home','/Users/xlx/Documents/proj/imgnet-flickr', ...
    'db_subdir', 'db2', 'exp_subdir', 'conceptrank-exp', 'topK', 15, ...
    'alph', .5, 'GradObj', 'on', 'solver', 'lbfgs', 'bfgs_ttits', 500, ...
    'obs_type', 'cn5-pr', 'init_type', 'cn4', 'entry_type', 'all') ;


[bigram, new_tagmap, new_tagcnt] = convert_syn_input(fullfile(exp_home, exp_subdir, in_file), ...
    'tagcnt_thresh', 3, 'numtag_thresh', 1200) ;

[cn4, cn_new, cn5] = conver_conceptnet(fullfile(exp_home, db_subdir, 'CN_graph.mat'), new_tagmap, 'logistic') ;

fprintf(1, '#tags %d, #bigrams %d\n', size(bigram,1), nnz(bigram));
fprintf(1, '\trecall of bigrams in cn-known %d (%0.4f)\n', nnz(bigram & cn4), nnz(bigram & cn4)/nnz(cn4));
fprintf(1, '\trecall of bigrams in cn-new %d (%0.4f)\n', nnz(bigram & cn_new), nnz(bigram & cn_new)/nnz(cn_new));

%bg1 = bigram.*(bigram>1) ;
%fprintf(1, '#bigrams>1 %d\n', nnz(bg1));
%fprintf(1, 'recall of bg1 in cn-known %d (%0.4f)\n', nnz(bg1 & cn4), nnz(bg1 & cn4)/nnz(cn4));
%fprintf(1, 'recall of bg1 in cn-new %d (%0.4f)\n', nnz(bg1 & cn_new), nnz(bg1 & cn_new)/nnz(cn_new));

[~, jnt] = sort(cell2mat(values(new_tagmap)) );
tag_list = keys(new_tagmap);
tag_list = tag_list(jnt);

[~] = eval_conceptrank(cn_new, bigram, cn4, 'verbose', 1);
print_top_pairs(tril(bigram), tag_list, topK, cn4, cn5, 'bigram') ;

if strcmp(obs_type, 'bigram')
    obsf = normalise(bigram);
else
    obsf = compute_pagerank_mat(cn5, alph);
end

alphR_val = 50/sum(bigram(:)>0) ; %[1e-9, 1e-6];%, 0.001 0.005 .01];% .02 .05 .1 .25 .5 1 2 4 10] ;
rG = .05*rand(size(cn4)) ;

for j = 1 %: length(alphR_val)
    aR = alphR_val(j) ;
    if strcmp(init_type, 'cn4')
        init_G = cn4 + rG ;
        lower_bound = cn4; 
    else
        init_G = cn5 + rG ;
        lower_bound = cn5; 
    end
    
    if strcmp(entry_type, 'bigram-only')
        init_G = init_G.*(bigram>0)  ;
    end
    fprintf(1, '\n%s alpha-R = %0.4f ... \n', datestr(now, 31), aR);
    for jm = 20 
        tic
        GW = opt_wrap_conceptrank(lower_bound, init_G, obsf, 'alphR', aR, 'alph', alph, ...
            'maxiter', jm, 'GradObj', gradobj, 'solver', solver, 'bfgs_ttits', bfgs_ttits);
        % bfgs_ttits is the max-total-its in LBFGS, default 5000
        toc
        [~] = eval_conceptrank(cn_new, GW+GW', cn4, 'verbose', 1, 'tau', 1e-6);
        init_G = GW ;
    end    
end

GW = GW+GW';
print_top_pairs(tril(GW), tag_list, topK, cn4, cn5, 'ConceptRank') ;

return




%% ----- function to convert wn input to matrixes -----
function [bigram, new_tagmap, new_tagcnt] = convert_syn_input(input_mat_name, varargin)
% optional params tagcnt_thresh, numtag_thresh)

load(input_mat_name) ;
%{
load('/Users/xlx/Documents/proj/imgnet-flickr/conceptrank-exp/syn_random10.mat')
whos
  Name                 Size             Bytes  Class    Attributes

  bigram_list      22356x1             178848  int64
  tag_list           458x12             10992  char
  tag_cnt
  usr_list            28x15               840  char               
  wnet_list           10x9                180  char    
%}

[tagcnt_thresh, numtag_thresh] = process_options(varargin, ...
    'tagcnt_thresh', 3, 'numtag_thresh', 1200);


c = double(reshape(bigram_list, 3, []));
c(1:2, :) = c(1:2, :) + 1; % zero-based indexing in python --> 1-based indexing here

% convert tag index
val_idx = unique( [ c(1, :), c(2, :) ] ) ;
tag_idx = find(tag_cnt >= tagcnt_thresh);
val_idx = intersect(tag_idx, val_idx);
    
if length(val_idx) > numtag_thresh % take top #numtag_thresh of tag
    [cntmp, jv] = sort(tag_cnt(val_idx), 'descend');
    cnt_thresh = cntmp(numtag_thresh);
    num_idx = find(tag_cnt > cnt_thresh); 
    val_idx = intersect(num_idx, val_idx); 
    %val_idx = val_idx(jv(1:numtag_thresh)) ;
end

nv = length(val_idx) ;
tagtmp = cellstr(tag_list) ;
new_tagmap = containers.Map(tagtmp(val_idx), num2cell(1:nv));
new_tagcnt = containers.Map(tagtmp(val_idx), num2cell(tag_cnt(val_idx)));

% construct new matrix index
[f1, f2] = deal(false(1, size(c, 2)));
for i = 1: size(c, 2)
    f1(i) = c(1,i)*any(c(1,i)==val_idx);
    f2(i) = c(2,i)*any(c(2,i)==val_idx);
end
c1 = c(1, f1>0 & f2>0); 
c2 = c(2, f1>0 & f2>0);
c3 = c(3, f1>0 & f2>0);
tmp_idxmap = containers.Map(num2cell(val_idx), num2cell(1:nv));
c1 = tmp_idxmap.values(num2cell(c1));
c2 = tmp_idxmap.values(num2cell(c2));
c1 = [c1{:}];
c2 = [c2{:}];

bigram = sparse(c1, c2, 1.*c3, nv, nv);

%% code to filter out insignificant words and bigram entries

bigram = bigram + bigram' ;
% cignore = c(1:2, c(3,:)<0); % index of the ignored positions

fprintf(1, '\n done loading data from %s \n', input_mat_name);
fprintf(1, '\n  pruned %d tags to %d (min cnt %d), %d bigram entries to %d \n', ...
    length(tag_list), nv, min(tag_cnt(val_idx)), size(bigram_list,1)/3, nnz(bigram)/2);

tk = keys(new_tagcnt);
tv = cell2mat(values(new_tagcnt));
[~, jt] = sort(tv, 'descend'); 
fprintf(1, ' most freq tags: \n');    fprintf(1, ' %s,', tk{jt(1:8)});
fprintf(1, '\n least freq tags: \n');   fprintf(1, ' %s,', tk{jt(end-7:end)}); fprintf(1, '\n\n');

%% ----- function to convert conceptnet graph into init-val + groundtruth for evaluation -----
function [cn_known, cn_new, cn_all] = conver_conceptnet(cn_graph_mat, tag_map, renorm_method)

load(cn_graph_mat, 'G4c', 'G5d', 'G5a', 'word_idmap' );
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
if nargin<3 || strcmpi(renorm_method, 'logistic')
    renorm_func = inline( '.5*(1-exp(-x))./(1+exp(-x))' ); % logistic function
else
    renorm_func = inline('log10(1+x)'); %logscaling
end

n = double(tag_map.Count) ;
nw = double(word_idmap.Count);
% map word_idmap to tag_map
wid_tid = zeros(n, 2); %[zeros(n, 1), (1:n)'];
tagkey = keys( tag_map );
tag_exist = isKey(word_idmap, tagkey);
for i = 1 : n
    if tag_exist(i)
        wid_tid(i, 1) = word_idmap(tagkey{i}) ;
    end
    wid_tid(i, 2) = tag_map(tagkey{i}) ;
end
[~, iw] = sort(wid_tid(:,2)) ;
wid_tid = wid_tid(iw, :); % the second column becomes 1:n
fprintf(1, ' reducing %d-word graph to %d tags, %d tags do not exist\n', word_idmap.Count, n, sum(~tag_exist) );

Gknown = sparse(nw, nw);
Gnew = sparse(nw, nw);
Gall = sparse(nw, nw);

nr = length(G4c);
assert(length(G5d)==nr, 'number of relatinos need to match');
for r = 1 : nr
    Gknown = Gknown + G4c{r} ;
    Gnew = Gnew + G5d{r} ;
    Gall = Gall + G5a{r} ;
end

fprintf(1, ' #relations in full matrix: %d in cn5-all, %d in cn4-core, %d new in cn5\n', nnz(Gall), nnz(Gknown), nnz(Gnew));
Gnew(Gknown>0) = 0;
fprintf(1, ' # exclusive relations (cn5-new - cn4-core) %d \n', nnz(Gnew));

wflag = wid_tid(:,1) & wid_tid(:,2);
iw1 = wid_tid(wflag,1);
iw2 = wid_tid(wflag,2);

[cn_known, cn_new, cn_all] = deal(zeros(n));
cn_known(iw2, iw2) = Gknown(iw1, iw1);
cn_new(iw2, iw2)   = Gnew(iw1, iw1);
cn_all(iw2, iw2)   = Gall(iw1, iw1);

cn_known = renorm_func(cn_known); 
cn_new = renorm_func(cn_new); 
cn_all = renorm_func(cn_all); 

fprintf(1, ' #relations in reduced matrix: %d in cn5-all, %d in cn4-core, %d new in cn5\n', nnz(cn_all), nnz(cn_known), nnz(cn_new));


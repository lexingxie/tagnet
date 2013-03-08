

[~,hostn] = system('hostname');
if strcmp(hostn(1:7), 'clavier') % macox
    exp_home = '/Users/xlx/Documents/proj/imgnet-flickr';
elseif strcmp(hostn(1:9), 'cantabile') % desktop
    exp_home = '/home/xlx/data/imgnet-flickr';
end

exp_subdir = 'conceptrank-exp' ;
dateid = datestr(now, 30);
log_name = sprintf('ilsvrc_eval_%s', dateid);
diary(fullfile(exp_home, exp_subdir, [log_name '.diary']));

synset_list = textread(fullfile(exp_home, exp_subdir, 'ilsvrc-intersect.txt'), '%s');

%logistic = inline( '.5*(1-exp(-x))./(1+exp(-x))' );

% this will just return a copy of the whole concept net, weights and indexes
[~, all_tag_map, CN4, CN_new, CN5] = learn_conceptrank('exp_home', exp_home, 'ops', 'load_conceptnet') ;
disp('');

assert(all(CN5(:,end)==0) && all(CN5(end,:)==0), 'last row/col of conceptnet mat is non zero!');
CN4 = CN4 + CN4'; CN_new=CN_new+CN_new'; CN5 = CN5 + CN5';
CN4 = CN4(1:end-1, 1:end-1);    CN_new = CN_new(1:end-1, 1:end-1);    CN5 = CN5(1:end-1, 1:end-1);

all_t = keys(all_tag_map);
all_idx = values(all_tag_map);  all_idx = [all_idx{:}]';
all_t = all_t(all_idx > 0);
all_idx = all_idx(all_idx > 0);
[all_idx, itmp] = sort(all_idx );
all_t = all_t(itmp);

nw = double(length(all_t)) ;
W = zeros(nw);
B = zeros(nw);

prec_depth = 1000;

syn_cnt = 0;

rand_idx = randperm(length(synset_list));
for k = 1 : length(synset_list)
    i = rand_idx(k); 
    in_mat_name = ['ilsvrc873/' synset_list{i} '.mat'] ;
    out_mat_name = ['ilsvrc873-out/' synset_list{i} '.mat'] ;
    out_mat_file = fullfile(exp_home, exp_subdir, out_mat_name);
    if ~exist(out_mat_file, 'file')
        continue;
        
    else
        [bigram, bg_list] = learn_conceptrank('exp_home', exp_home, 'ops', 'load_bigram', 'in_file', in_mat_name) ;
        % match vocabularies 
        [tk, ik, jk] = intersect(bg_list, all_t);
        tv = 1 : length(tk) ;
        bg = .5*full(bigram + bigram');
        B(jk, jk) = B(jk, jk) + bg(ik, ik);
        
        % [GW, tag_list, cn4, cn_new, cn5]
        load(out_mat_file, 'GW', 'tag_list', 'cn_new', 'cn5');
        assert(length(tag_list)==size(GW, 1), 'tag and weight matrix dimension disagree!');
        
        %%% handle vocabulary mismatch between conceptnet and bigrams
        [tk, ik, jk] = intersect(tag_list, all_t);
        %tv = 1 : length(tk) ;
        
        g = full(GW + GW');
        W(jk, jk) = W(jk, jk) + g(ik, ik);
        
        syn_cnt = syn_cnt + 1;
    end
    if mod(syn_cnt, 25) == 0 || k == length(synset_list)
        jz = nw - sum(sum(W, 1) == 0);        
        
        fprintf(1, '%s processed %d synsets, %d of %d concepts covered \n', datestr(now,31), syn_cnt, jz, nw);
        
        rperf = eval_conceptrank(CN_new, W, CN4, 'prec_depth', prec_depth, 'nrounds', 1, 'tau', 0, 'verbose', 0) ;
        fprintf(1, '%s W  num-rel %d, \tprior: %0.4f, \tmax-recall: %0.4f, \tprior-covered: %0.4f \n', ...
            datestr(now,31), rperf.npos, rperf.prior, rperf.max_recall, rperf.r_prior);
        fprintf(1, '%s W  \tAP: %0.4f, \tF1: %0.4f, \tP@%d: %0.4f\n', ...
            datestr(now,31), rperf.ap(1), rperf.f1(1), prec_depth, rperf.p_at_d(1));
        
        bperf = eval_conceptrank(CN_new, B, CN4, 'prec_depth', prec_depth, 'nrounds', 1, 'tau', 0, 'verbose', 0) ;
        fprintf(1, '%s BG \tAP: %0.4f, \tF1: %0.4f, \tP@%d: %0.4f\n\n', ...
            datestr(now,31), bperf.ap(1), bperf.f1(1), prec_depth, bperf.p_at_d(1));
        
        %print_top_pairs(triu(W, 1), all_t, 50, triu(CN4, 1), triu(CN5, 1), 'ConceptRank') ;
        % rperf.top_returns = [ j1, j2, rs(1:prec_depth), tlabel(itmp) ];
        fprintf(1, ' top-rel: new-cn5, score, w1--w2 \n');
        for j = 1 : 10
            dj = rperf.top_returns(j, :);
            fprintf(1, '\t %d \t %3.3f \t %s -- %s \n', dj(4)>0, dj(3), all_t{dj(1)}, all_t{dj(2)}  ) ;
        end
        
        if any(syn_cnt == [25 50 100 200 400])
            mperf = eval_conceptrank(CN_new, W, CN4, 'prec_depth', 10, 'eval_mode', 'macro', 'verbose', 0) ; 
            bmperf = eval_conceptrank(CN_new, B, CN4, 'prec_depth', 10, 'eval_mode', 'macro', 'verbose', 0) ; 
            mat_name = fullfile(exp_home, exp_subdir, sprintf('%s%_n%03d.mat', log_name, syn_cnt) );
            save(mat_name)
        end
    end
    
end

diary off;

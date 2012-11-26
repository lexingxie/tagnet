

clear

exp_envsetup

whos

%pr_matname = 'CN5_prnorm.mat';
pg_matname = 'cn_learned_avg.mat' ;
%   eff_new_tag_list         1x3109              387062  cell                
%   norm_eff_cn           3109x3109            77327048  double   
pr_matname = 'cn_rank_avg.mat' ;
%   outpr      3109x3109            77327048  double  

pr_graph_mat = fullfile(data_dir, '../conceptrank-exp/syn93_output', pg_matname);
pr_rank_mat = fullfile(data_dir, '../conceptrank-exp/syn93_output', pr_matname);

Ycache_mat = fullfile(data_dir, ['Y_' pr_matname]);


tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore', 'target_tags');
%   found_wn            81x1                     81  logical
%   synset_map       13288x1                    112  containers.Map
%   tag_feat            63x7323              765088  double            sparse
%   target_tags         81x1                   9972  cell
%   vocab             7323x1                 915782  cell
%   vscore               1x7323               58584  double
    
NUMV = 500;

if strcmp(hostn(1:7), 'clavier') % macox
    
    [vs, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
    tag_feat = tag_feat(:, iv(1: NUMV));
    Y = log(tag_feat + 1)';
    Y = Y/max(Y(:));
    
    %save(pr_graph_mat, 'G5', 'G5p', 'word_idmap', 'alph');
    load(pr_graph_mat, 'eff_new_tag_list');
    num_tag = length(eff_new_tag_list);
    word_idmap = containers.Map(eff_new_tag_list, num2cell(1:num_tag));
    
    load(pr_rank_mat, 'outpr');
    
    [nrow, ncol] = size(Y) ;
    gp_col = target_tags(found_wn);
    gp_row = vocab(iv(1: NUMV));
    gp_col_id = zeros(1, ncol) ;
    gp_row_id = zeros(1, nrow) ;
    [col_err, row_err] = deal([]);
    
    for i = 1 : ncol
        if isKey(word_idmap, gp_col{i})
            gp_col_id(i) = word_idmap( gp_col{i} );
        elseif isKey(word_idmap, gp_col{i}(1:end-1))
            gp_col_id(i) = word_idmap( gp_col{i}(1:end-1) );
        elseif strcmp(gp_col{i}, 'running')
            gp_col_id(i) = word_idmap('run');
        elseif strcmp(gp_col{i}, 'dancing')
            gp_col_id(i) = word_idmap('dance');
        elseif strcmp(gp_col{i}, 'buildings')
            gp_col_id(i) = word_idmap('build');
        else
            fprintf(1, 'col key #%d not found "%s"!\n', i, gp_col{i});
            gp_col_id(i) = 1;
            col_err = [col_err i];
        end
    end
    for i = 1 : nrow
        if isKey(word_idmap, gp_row{i})
            gp_row_id(i) = word_idmap( gp_row{i} );
        elseif isKey(word_idmap, gp_row{i}(1:end-1))
            gp_row_id(i) = word_idmap( gp_row{i}(1:end-1) );
        else
            fprintf(1, 'row key #%d not found "%s"!\n', i, gp_row{i});
            gp_row_id(i) = -1;
            row_err = [row_err i];
        end
    end
    gp_rr_id = gp_row_id(gp_row_id>0); % remove rows that are not in word_idmap
    
    G5p = outpr + outpr' ;
    G5p = G5p - diag(diag(G5p)); % remove diagonal entry
    
    col_valid = setdiff(1:ncol, col_err);
    
    gp_row_id(gp_row_id<0) = 1;
    Yadd = G5p(gp_row_id, gp_col_id);
    if ~isempty(col_err)
        Yadd(:, col_err) = mean(Yadd(:,col_valid), 2);
    end
    Yadd = Yadd/max(Yadd(:)) + Y ;
    Yadd(row_err, :) = Y(row_err, :);
    
    Y5p = G5p(gp_rr_id, gp_col_id) ;
    if ~isempty(col_err)
        Y5p(:, col_err) = mean(Y5p(:,col_valid), 2);
    end
    Y5p = Y5p/max(Y5p(:));
    
    save(Ycache_mat, 'Y5p', 'Y', 'Yadd');
else
    fprintf(1, 'quit: do not have input data on this machine!\n');
end



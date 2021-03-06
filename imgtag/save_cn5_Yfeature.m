
%eval_str = 'eval_TestCN5pr_10k_' ;
clear

exp_envsetup
%exp_setparams

whos

pr_matname = 'CN5_prnorm.mat';
%pr_matname = 'CN5_pr.mat' ;
Ycache_mat = fullfile(data_dir, ['Y_' pr_matname]);
pr_graph_mat = fullfile(data_dir, '../db2/conceptnet-graph', pr_matname);

tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore', 'target_tags');
    
if strcmp(hostn(1:7), 'clavier') % macox
    %% load tag features, setup Y
    
    %   found_wn            81x1                     81  logical
    %   synset_map       13288x1                    112  containers.Map
    %   tag_feat            63x7323              765088  double            sparse
    %   target_tags         81x1                   9972  cell
    %   vocab             7323x1                 915782  cell
    %   vscore               1x7323               58584  double
    
    NUMV = 500;
    
    [vs, iv] = sort(vscore, 'descend'); % take ~150 dimensions for now
    tag_feat = tag_feat(:, iv(1: NUMV));
    Y = log(tag_feat + 1)';
    Y = Y/max(Y(:));
    
    %save(pr_graph_mat, 'G5', 'G5p', 'word_idmap', 'alph');
    load(pr_graph_mat, 'G5', 'G5p', 'word_idmap');
    
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
    gp_rr_id = gp_row_id(gp_row_id>0);
    
    G5p = G5p + G5p' ;
    %G5p = G5p - diag(diag(G5p)); % remove diagonal entry
    
    col_valid = setdiff(1:ncol, col_err);
    
    gp_row_id(gp_row_id<0) = 1;
    Yadd = G5p(gp_row_id, gp_col_id);
    for e = col_err
        Yadd(:, e) = mean(Yadd(:,col_valid), 2);
    end
    %Yadd(:, col_err) = 0;
    Yadd = Yadd/max(Yadd(:)) + Y ;
    Yadd(row_err, :) = Y(row_err, :);
    
    Y5p = G5p(gp_rr_id, gp_col_id) ;
    %Y5p(:, col_err) = 0;
    for e = col_err
        Y5p(:, e) = mean(Y5p(:,col_valid), 2);
    end
    Y5p = Y5p/max(Y5p(:));
    
    % produce the "add" and standalone version 
    % of the original graph feature
    rawg = G5 ;
    %rawg = rawg - diag(diag(rawg)); 
    % do not remove diagonal entry (helps similarity)
    
    Yadg = rawg(gp_row_id, gp_col_id);
    for e = col_err
        Yadg(:, e) = mean(Yadg(:,col_valid), 2);
    end
    Yadg = Yadg/max(Yadg(:)) + Y ;
    Yadg(row_err, :) = Y(row_err, :); % fill in empty row with Y
    
    Y5g = rawg(gp_rr_id, gp_col_id) ;
    for e = col_err
        Y5g(:, e) = mean(Y5g(:,col_valid), 2);
    end
    Y5g = Y5g/max(Y5g(:));
    
    save(Ycache_mat, 'Y5p', 'Y', 'Yadd', 'Y5g', 'Yadg');
else
    fprintf(1, 'quit: do not have input data on this machine!\n');
end



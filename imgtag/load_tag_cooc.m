
function load_tag_cooc(col_vocab_file, out_mat)

%% -----------
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    db_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
else % mac os
    db_dir = '/Users/xlx/proj/ImageNet/db2';
end


% load data
wv_mat = fullfile(db_dir, 'wordnet_tag_stat.mat');
load(wv_mat, 'wvmat', 'vocab', 'wnlist', 'vcnt', 'vscore');
[~, iv] = sort(vscore, 'descend');
row_label = vocab(iv);
wvmat = wvmat(:, iv);

wnet_tag_mat = fullfile(db_dir, 'wnet_tag_map.mat');
load(wnet_tag_mat, 'tag_map');

col_label = textread(col_vocab_file, '%s');

%% read bigram data
bg_file = fullfile(db_dir, 'flickr.bigram.ge5.txt');

%col_label = textread(fullfile(data_dir, 'vocab_sun09.txt'), '%s');
%tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
%load(wv_mat, 'vocab', 'vcnt', 'vscore');
% make the dimensions sort by count

% read bigram file
BG = load_bigram_list(bg_file, row_label, col_label);
nzb = find(sum(BG, 2)>0) ;
BG = BG(nzb, :);
row_label_BG = row_label(nzb);
fprintf(1, 'mapped %dx%d bigram matrix to %dx%d, %d values, non-zero ratio %0.2f\n', ...
    size(wvmat), size(BG), nnz(BG), nnz(BG)/numel(BG) );

%% read wordnet cooccurrence, map to tags

wnet_id_map = containers.Map(wnlist, num2cell(1:length(wnlist)));
nc = length(col_label);
nr = size(wvmat, 2);
val = cell(nc, 1);
for i = 1 : nc
    if isKey(tag_map, col_label{i})
        ww = tag_map(col_label{i}) ; % list of wnet ids for a tag
        cur_cnt = zeros(nr, 1);
        for j = 1 : length(ww)
            jw = wnet_id_map(ww{j}) ;
            cur_cnt = cur_cnt + full(wvmat(jw, :))' ;
        end
        ii = find(cur_cnt); % all non-zero entries for the current tag
        val{i} = [ii, i*(ii>0), cur_cnt(ii)] ;
        %fprintf(1, ' tag "%s", %d non-zero values\n', col_label{i}, length(ii));
    else
        fprintf(1, ' NOT FOUND, col %d tag "%s", fill with BG \n', i, col_label{i});
        ii = find(BG(:, i));
        val{i} = [ii, i*(ii>0), BG(ii, i)] ;
    end
end
val = cat(1, val{:});
WG = sparse(val(:, 1), val(:,2), val(:,3), nr, nc, size(val,1));
nzw = find(sum(WG, 2)>0) ;
WG = WG(nzw, :);
row_label_WG = row_label(nzw);

fprintf(1, 'mapped %dx%d wnet-tag matrix to %dx%d, %d values, non-zero ratio %0.2f\n', ...
    size(wvmat), size(WG), size(val,1), size(val,1)/numel(WG) );


save(out_mat, 'BG', 'WG', 'row_label*', 'col_label');

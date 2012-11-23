

%% -----------
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
elseif strcmp(hostn, 'cantabile')
    data_dir = '~/Data/imgnet-flickr/db2';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/db2';
end


%% load wordnet-tag correspondence
out_mat = fullfile(data_dir, 'wordnet_tag_stat.mat');
if exist(out_mat, 'file')
    load(out_mat);  % var called 'wvmat', 'vocab', 'wnlist'
else
    wn_stat_file = fullfile(data_dir, 'wordnet_tagsfreq.txt');
    vocab_file = fullfile(data_dir, 'vocab_flickr.txt');
    unigram_file = fullfile(data_dir, 'flickr.unigram.txt');
    wnet_list_file = fullfile(data_dir, 'wnet-50.txt');
    
    [synset_count, synset_list] = textread(wnet_list_file, '%d%s');
    synset_map = containers.Map(synset_list, num2cell(synset_count));
    
    vocab = sort(textread(vocab_file, '%s'));
    vcnt = get_vocab_counts(vocab, unigram_file);    
    [wvmat, wnlist] = read_wn_vocab_stat(wn_stat_file, vocab);
    vscore = score_vocab_dimensions(wvmat, synset_map, wnlist);
    
    save(out_mat, 'wvmat', 'vocab', 'wnlist', 'vcnt', 'vscore', '*_file');
end

%% load bigram

bg_mat = fullfile(data_dir, 'bg_data.mat');

if exist(bg_mat, 'file')
    load(bg_mat, 'BG', 'vdim', 'vocab_map');  
else
    bg_file = fullfile(data_dir, 'flickr.bigram.ge5.txt');
    col_label = textread(fullfile(data_dir, 'vocab_sun09.txt'), '%s');
    %tag_feat_mat = fullfile(data_dir, 'tag_wn_feature.mat');
    load(out_mat, 'vocab', 'vcnt', 'vscore');
    % make the dimensions sort by count
    [sv, iv] = sort(vscore, 'descend');
    
    row_label = vocab(iv);
    
    % read bigram file
    BG = load_bigram_list(bg_file, vdim, col_label);
    save(bg_mat, 'BG', 'row_label', 'col_label'); 
end
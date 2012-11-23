
function [out_vocab, flag_t, tag_map] = normalize_vocab(in_tag_file, db_dir)

addpath ../mksqlite-1.11-src/

if nargin < 2
    [~,hostn] = system('hostname');
    hostn = deblank(hostn);
    if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
        %data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/sun09';
        db_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
    else % mac os
        db_dir = '/Users/xlx/proj/ImageNet/db2';
    end
end

% map tags
sql_dict = fullfile(db_dir, 'dict.db');
mksqlite('open', sql_dict);
res = mksqlite('SELECT string,word FROM dict');
mksqlite('close');
str_map = containers.Map({res.string}, {res.word});
addl_words = textread(fullfile(db_dir, 'places_etc.txt'), '%s');
str_map = [containers.Map(addl_words, addl_words); str_map];

in_tag = textread(in_tag_file, '%s');

flag_t = isKey(str_map, in_tag) ;

out_vocab = values(str_map, in_tag(flag_t)) ;


tag_map = [containers.Map(in_tag(flag_t), out_vocab); containers.Map(in_tag(~flag_t), repmat({''}, 1, sum(~flag_t)) )];

out_vocab = unique(out_vocab);
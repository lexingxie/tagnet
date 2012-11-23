[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/db2';
end


tag_feat_mat = fullfile(data_dir, 'wordnet_tag_stat.mat');
%load(tag_feat_mat, 'tag_feat', 'found_wn', 'vocab', 'vcnt', 'vscore');

load(tag_feat_mat, 'wvmat', 'vocab', 'wnlist', 'vcnt', 'vscore');

fo = fopen(fullfile(data_dir, 'flickr_vscore.txt'), 'wt');

nv = length(vocab);
[~, jv] = sort(vscore);
vsprct(jv) = (1 : nv)/nv ;

for i = 1 : nv
	fprintf(fo, '%s\t%0.4f\t%0.4f\n', vocab{i}, vscore(i), vsprct(i));
end

fclose(fo);
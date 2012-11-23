
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/db';
end


%% load bigram
bg_mat = fullfile(data_dir, 'bg_data.mat');

if exist(bg_mat, 'file')
    load(bg_mat, 'BG', 'vdim', 'vocab', 'vcnt', 'vocab_map', 'nv');  
else
    bg_file = fullfile(data_dir, 'bigram_filtered.txt');
    
    tag_feat_mat = fullfile(data_dir, '../nuswide', 'tag_wn_feature.mat');
    load(tag_feat_mat, 'vocab', 'vcnt', 'vscore');
    % make the dimensions sort by count
    [sv, iv] = sort(vcnt, 'descend');
    vocab_map = containers.Map(vocab(iv), num2cell(1:length(iv)));
    nv = double(vocab_map.Count);
    vdim = vocab(iv);
    
    % read bigram file
    BG = load_bigram_list(bg_file, vocab_map);
    [ii, jj] = find(BG);
    xxv = zeros(1, nv);
    xxu = zeros(1, nv);
    for v = 1 : nv
        jv = jj(ii==v);
        xv = full(BG(v, jv));
        xxv(v) = sqrt(xv(:)'*xv(:));
        ju = ii(jj==v);
        xu = full(BG(ju, v));
        xxu(v) = sqrt(xu(:)'*xu(:));
        if mod(v,1000)==0
            fprintf(1, '\t factor v=%d\n', v);
        end
    end
    BGn = BG;
    for v = 1 : nv
        jv = jj(ii==v);
        ju = ii(jj==v);
        if xxv>0
            BGn(v, jv) = BG(v, jv)/sqrt(xxv); % normalize row
        end
        if xxu>0
            BGn(ju, v) = BG(ju, v)/sqrt(xxu); % normalize col
        end
        if mod(v,1000)==0
            fprintf(1, '\t normalized v=%d\n', v);
        end
    end
    save(bg_mat, 'BG', 'BGn', 'vdim', 'vocab', 'vcnt', 'vocab_map', 'nv'); 
end

%% load conceptnet, compute similarity
cn_mat = fullfile(data_dir, 'cn_data.mat');

if exist(cn_mat, 'file')
    load(cn_mat, 'sim');  
else
    load ( fullfile(data_dir, 'conceptnet_sim_U.txt') )
    load ( fullfile(data_dir, 'conceptnet_sim_S.txt') )
    rowlabels = textread(fullfile(data_dir, 'conceptnet_sim_rowlabel.txt'), '%s');
    % -- dimensions order different, map to vocab_map first
    vc = cell2mat(values(vocab_map, rowlabels) );
    conceptnet_sim_U = conceptnet_sim_U(vc, :);
    
    ss = conceptnet_sim_U*diag(conceptnet_sim_S)*conceptnet_sim_U';
    [ii, jj] = find(abs(ss<1e3));
    sim = sparse(ii, jj, ss(sub2ind([nv nv], ii, jj)) );
    
    clear ss
    save(cn_mat, 'sim');  
end

%% sort and find out surprising tag pairs
ib = find(tril(BG));
nb = length(ib);
[~, ii] = sort(full(BG(ib)), 'descend');
[~, jj] = sort(full(sim(ib)), 'descend');

B_rk(ii) = 100*(nb:-1:1)/nb;   % prctile rank from the bottom
S_rk(jj) = 100*(nb:-1:1)/nb;   % B_rk(i) correspond to the rank of BG(ib(i))

regionA = B_rk + S_rk;     % B++, S++
regionB = S_rk - B_rk;     % B--, S++
regionC = B_rk - S_rk;     % B++, S--
regionD = -B_rk - S_rk;    % B--, S--
[~, ja] = sort(regionA, 'descend');  % higher rank in both B and S is better
[~, jb] = sort(regionB, 'descend');  % lower rank in B, higher rank in S
[~, jc] = sort(regionC, 'descend');  % higher rank in both B and S is better
[~, jd] = sort(regionD, 'descend');  % lower rank in B, higher rank in S

topN = 50;
[pi, pj] = ind2sub([nv nv], ib(ja(1:topN)));
word_pair_A = [vdim(pi), vdim(pj)] ;
[pi, pj] = ind2sub([nv nv], ib(jb(1:topN)));
word_pair_B = [vdim(pi), vdim(pj)] ;
[pi, pj] = ind2sub([nv nv], ib(jc(1:topN)));
word_pair_C = [vdim(pi), vdim(pj)] ;
[pi, pj] = ind2sub([nv nv], ib(jd(1:topN)));
word_pair_D = [vdim(pi), vdim(pj)] ;
fprintf(1, '\n\nrA#\tB%%\tS%%\tword1\tword2\n');
for j = 1: 30
    fprintf(1, '%3d\t%3.2f%%\t%3.2f%%\t%s\t%s\n', j, B_rk(ja(j)), S_rk(ja(j)), word_pair_A{j, :});
end

fprintf(1, '\n\nrB#\tB%%\tS%%\tword1\tword2\n');
for j = 1: 20
    fprintf(1, '%3d\t%3.2f%%\t%3.2f%%\t%s\t%s\n', j, B_rk(jb(j)), S_rk(jb(j)), word_pair_B{j, :});
end

fprintf(1, '\n\nrC#\tB%%\tS%%\tword1\tword2\n');
for j = 1: 20
    fprintf(1, '%3d\t%3.2f%%\t%3.2f%%\t%s\t%s\n', j, B_rk(jb(j)), S_rk(jb(j)), word_pair_C{j, :});
end

fprintf(1, '\n\nrD#\tB%%\tS%%\tword1\tword2\n');
for j = 1: 20
    fprintf(1, '%3d\t%3.2f%%\t%3.2f%%\t%s\t%s\n', j, B_rk(jb(j)), S_rk(jb(j)), word_pair_D{j, :});
end
%% compare the two matrixes

icp = find(tril(sim>0));
icn = find(tril(sim<0));
idxp = intersect(ib, icp);
idxn = intersect(ib, icn);

nz_Bp = log10(full(BG(idxp)));
nz_Bn = log10(full(BG(idxn)));
nz_Sp = log10(full(sim(idxp)));
nz_Sn = log10(-full(sim(idxn)));

if 0
    [hbp, xb] = hist(nz_Bp, 100);
    hbn = hist(nz_Bp, 100, xb);
    [hsp, xs] = hist(nz_Sp, 100);
    hsn = hist(nz_Sn, 100, xs);
    %figure; bar(xb, [hbp; -hbn]');
    %figure; bar(xs, [hsp; -hsn]');
end
% plot bubble of significance
figure; plot(sort([nz_Bp, nz_Sp], 'descend'), '.', 'markersize', 3); axis tight; grid on
figure; plot(nz_Sp, nz_Bp, '.', 'markersize', 3); axis tight; grid on
figure; plot(nz_Sn, nz_Bn, 'r.', 'markersize', 3); axis tight; grid on
set(gca, 'xticklabel', [], 'yticklabel', [])

%%
% figure out the percentile of both lists
% [vs, iv] = sort(vcnt, 'descend');
% vlabel = vocab(iv);
% vfprct(iv) = 1 - (0 : nv-1)/nv ;
% [~, jv] = sort(vscore);
% vsprct(jv) = (1 : nv)/nv ;
npos = length(nz_Bp);
[sbp, ii] = sort(nz_Bp, 'descend');
Bp_rank(ii) = 1:npos;
[~, ii] = sort(nz_Sp, 'descend');
Sp_rank(ii) = 1:npos;

nneg = length(nz_Bn);
[~, ii] = sort(nz_Bn, 'descend');
Bn_rank(ii) = 1:nneg;
[~, ii] = sort(nz_Sp, 'descend');
Sn_rank(ii) = 1:nneg;

%TH_Bp = prctile(nz_Bp, [1 99]);
%TH_Sp = prctile(nz_Sp, [1 99]);

%topN = round(npos*0.01);
%ii = find(Bp_rank<=topN & Sp_rank<=topN);
[sr, ir] = sort(Bp_rank + Sp_rank) ;
[pi, pj] = ind2sub([nv nv], idxp(ir(1:20)));
word_pair_p = [vdim(pi), vdim(pj)] ;

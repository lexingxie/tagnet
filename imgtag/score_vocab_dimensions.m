function vscore = score_vocab_dimensions(wvmat, synset_map, wnlist)

[~, nv] = size(wvmat);
% pw = zeros(nw, 1);
% for w = 1 : nw
%     pw(w) = synset_map(wnlist{w});
% end
pw = cell2mat( values(synset_map, wnlist) );
pw = pw / sum(pw);


vscore = zeros(1, nv);
for v = 1 : nv
    pv = full(wvmat(:, v));
    pv ( pv < .5 ) = .5;
    pv = pv / sum(pv);
    vscore(v) = pw' * log2(pw ./ pv);
end
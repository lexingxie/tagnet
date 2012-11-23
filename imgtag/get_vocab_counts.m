
function vcnt = get_vocab_counts(vocab, unigram_file)

try
    [unig, unic] = textread(unigram_file, '%s%d');
catch
    [unic, unig] = textread(unigram_file, '%d%s');
end

vcnt = zeros(length(vocab),1);

for v = 1 : length(vocab)
    tf = strcmp(vocab{v}, unig);
    assert(sum(tf)==1, 'there should be one and only one match!');
    vcnt(v) = unic(tf);
end
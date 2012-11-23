
function BG = load_bigram_list(bg_file, row_label, col_label)

if nargin < 3
    col_label = [];
end

rdict = containers.Map(row_label, num2cell((1:length(row_label))') );

if isempty(col_label)
    cdict = rdict;
else
    cdict = containers.Map(col_label, num2cell((1:length(col_label))') );
end

nr = double(rdict.Count);
nc = double(cdict.Count);

fh = fopen(bg_file, 'r');

tmp = textscan(fh, '%d%s%s');
[bcnt, word1, word2] = deal(tmp{:});
% line format:
%     15   aardvark        zoo
%     5   abacus  acanthus
%     5   abacus  arcade
%     8   abacus  benedictine
%    11   abacus  blue
fclose(fh);

v1 = isKey(rdict, word1);
v2 = isKey(cdict, word2);

w1 = cell2mat(values(rdict, word1(v1 & v2)) );
w2 = cell2mat(values(cdict, word2(v1 & v2)) );
cc = double(bcnt (v1 & v2));

if isempty(col_label) 
    % symmetric square sparse matrix
    BG = sparse([w1;w2], [w2;w1], [cc;cc], nr, nc);
else
    v3 = isKey(rdict, word2);
    v4 = isKey(cdict, word1);
    
    w3 = cell2mat(values(rdict, word2(v3 & v4)) );
    w4 = cell2mat(values(cdict, word1(v3 & v4)) );

    c3 = double(bcnt (v3 & v4));
    BG = sparse([w1;w3], [w2;w4], [cc;c3], nr, nc);
end

fprintf(1, 'load %d x %d bigram matrix with %d entries, non-zero ratio %0.2f \n', ...
    nr, nc, nnz(BG), 2*length(w1)/(nr*nc) );
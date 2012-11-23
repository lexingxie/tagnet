function [wvmat, wnlist] = read_wn_vocab_stat(in_stat_file, vocab)

nv = length(vocab);
vocab_map = containers.Map(vocab, num2cell( (1:nv)' ) );

[~, result] = system( ['wc -l ', in_stat_file] );
nlines = sscanf(result, '%d', 1);

wnlist = cell(nlines, 1);
wvmat = sparse(nlines, nv); %, nlines*300);
cnt = 0;
fi = fopen(in_stat_file, 'r');
while ~feof(fi)
    tline = fgetl(fi);
    % lines look like the following
    % n15102894	wood:126 tree:123 fence:69 hole:60 texture:44 bark:39 ...
    cnt = cnt + 1;
    tmp = textscan(tline,'%s', 'delimiter', '\t :','multipleDelimsAsOne',1);
    wnlist{cnt} = tmp{1}{1};
    dcell = tmp{1}(2:end); % data
    wkeys = dcell(1:2:end);
    wval = str2num(char(dcell(2:2:end)')); %#ok<ST2NM>
    
    widx = cell2mat( values(vocab_map, wkeys) );
    wvmat(cnt, widx) = wval ;
    
    if mod(cnt, 100)==0
        fprintf(1, '%s %5d synsets loaded, %d words total\n', datestr(now, 31), cnt, nnz(wvmat));
    end
end
%fprintf(' last row: %s', wnlist{cnt});
%disp(wkeys(:)');
%disp(widx(:)')
%disp(wval(:)')

fprintf(1, '%s %5d synsets loaded, %d words total\n', datestr(now, 31), cnt, nnz(wvmat));
fclose(fi);
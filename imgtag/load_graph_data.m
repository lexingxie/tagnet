function [G, Ginfo, Grl] = load_graph_data(in_file, row_label, col_label, row_reduce, max_num_slices)

if nargin < 3
    col_label = [];
end
if nargin < 4
    row_reduce = false;
end
if nargin < 5
    max_num_slices = inf;
end

rdict = containers.Map(row_label, num2cell((1:length(row_label))') );

if isempty(col_label)
    cdict = rdict;
else
    cdict = containers.Map(col_label, num2cell((1:length(col_label))') );
end

fh = fopen(in_file, 'rt');
gn = 0;
gcnt = 0;
G = {};
Grl = {};
while ~feof(fh) 
    gn = gn + 1;
    [g, gi] = load_one_graph(fh, rdict, cdict);
    if row_reduce
        [g, rr] = reduce_graph(g, row_label, col_label);
    else
        rr = {};
    end
    [i,j,~] = find(g);
    more_than_one_term = length(unique(j))>=2 && length(unique(i))>=2 ;
    if ~isempty(g) && nnz(g)>2 && more_than_one_term
        gcnt = gcnt + 1;
        [G{gcnt}, Ginfo{gcnt}] = deal(g, gi);
        if ~isempty(rr)
            Grl{gcnt} = rr;
        end
        fprintf(1, ' read graph #%d, %d-th valid, %d edges, %d skipped\n', gn, gcnt, nnz(G{gcnt}), Ginfo{gcnt}{4}(2) );
        if gcnt >= max_num_slices
            break;
        end
    else
        fprintf(1, ' skip graph #%d, skip\n', gn);
    end
end

% G = cat(3, G{:});  % cannot cat sparse mat this way
Ginfo = cat(1, Ginfo{:});

fclose(fh);

%% --------------
function [gnew, rr] = reduce_graph(g, row_label, col_label)

[i0,j0,v0] = find(g);
%nr = length(row_label);

rset = unique(i0);
mlabel = setdiff(row_label(rset), col_label) ;
rr = [col_label; mlabel];
i1 = zeros(size(i0));
for jr = 1 : length(rr)
    jj = strmatch(rr{jr}, row_label(i0), 'exact');
    i1(jj) = jr;
end
%ii = find(setdiff(1:nr, [unique(i)
gnew = sparse(i1, j0, v0, length(rr), length(col_label), length(v0));


%% --------------
function [g, ginfo] = load_one_graph(fh, rdict, cdict)

cur_line = fgetl(fh);
ginfo = {};
glist = {};
ecnt = 0;
scnt = 0;

while ~feof(fh) 
    if length(cur_line)>3 && strcmp(cur_line(1:3), '---')
        break
    elseif length(cur_line)<1
        cur_line = fgetl(fh);
        continue; % skip blank line
    end
    %if ~isempty(regexp(cur_line, '^\s*$', 'once'))
    %    continue % skip blank line
    %end
    if cur_line(1)=='#'
        [~, remain] = strtok(cur_line(2:end), ',');
        [ginfo{1}, remain] = strtok(remain, ',');
        [ginfo{2}, remain] = strtok(remain, ',');
        ginfo{3} = remain ;
    elseif uint8(cur_line(1))>= uint8('0') && uint8(cur_line(1))<= uint8('9') % is a digit
        tmp = textscan(cur_line, '%f%s%s');
        if isKey(rdict, tmp{2}{1}) && isKey(cdict, tmp{3}{1})
            i1 = rdict(tmp{2}{1}) ;
            i2 = cdict(tmp{3}{1}) ;
            ecnt = ecnt + 1;
            glist{ecnt} = [i1, i2, tmp{1}];
        else
            scnt = scnt + 1;
        end
    end    
    cur_line = fgetl(fh);
end
ginfo{4} = [ecnt, scnt];

if ecnt >= 1
    glist = cat(1, glist{:});
    g = sparse(glist(:,1), glist(:,2), glist(:,3), length(rdict), length(cdict));
else
    g = [];
end


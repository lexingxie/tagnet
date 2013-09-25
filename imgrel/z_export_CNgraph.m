
load /Users/xlx/Documents/proj/imgnet-flickr/db2/CN_graph.mat
proj_dir = '/Users/xlx/Documents/proj/relearn/db';

% whos
%   Name               Size              Bytes  Class             Attributes
% 
%   G4c                1x14            1701088  cell                        
%   G5a                1x14            1981568  cell                        
%   G5d                1x14            1156608  cell                        
%   rel_idmap         14x1                 112  containers.Map              
%   word_idmap      7797x1                 112  containers.Map    

G = G4c; 
fh = fopen( fullfile(proj_dir, 'G4_core.txt'), 'wt') ;
fprintf(1, '%s exporting %d entries from graph\n', datestr(now, 31), sum(cellfun(@nnz, G4c)) );

rtxt = sort( keys(rel_idmap) );
ww = keys(word_idmap);
wj = values(word_idmap);    wj = cat(1, wj{:});
for r = 1 : length(rtxt)
    ridx = rel_idmap( rtxt{r} );
    [row, col] = find( G{r} );
    for j = 1 : length(row)
        rj = find(wj == row(j)) ;
        cj = find(wj == col(j)) ;
        fprintf(fh, '%s,%s,%s\n', rtxt{r}, ww{rj}, ww{cj});
    end
    fprintf(1, '%s wrote %d entries for relation "%s"\n', datestr(now, 31), length(row), rtxt{r} );
end

fclose(fh);
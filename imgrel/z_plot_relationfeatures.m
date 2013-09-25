
clf;

data_dir = '/Users/xlx/Documents/proj/relearn/tmp';
%mat_name = 'frame_word.2013-05-25_11h12m58.mat';
mat_name = 'frame_feat.2013-05-25_11h41m39.mat';

load(fullfile(data_dir, mat_name));
% savemat(out_mat, {"rdim":rdim, "wdim":wdim, 'm':mat})

logm = log10(m + 1);

imagesc(logm);
colorbar; 

rdim = cellstr(rdim);
wdim = cellstr(wdim);

axy = get(gca, 'position');
set(gca, 'position', axy + [0 0.12 .05 -.1] )
set(gca, 'ytick', 1:length(rdim), 'yticklabel', rdim);
set(gca, 'xtick', [])
yw = length(rdim) + .6 ;
nw = length(wdim) ;
text( 1:nw, yw*ones(nw,1), wdim, 'rotation', 90, 'horizontalalignment', 'right', 'interpreter', 'none') ;

%text(3, 15, 'testtest', 'rotation', 90) ;

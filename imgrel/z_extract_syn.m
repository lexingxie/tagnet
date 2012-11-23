
exp_home = '/Users/xlx/Documents/proj/imgnet-flickr/conceptrank-exp';

c1 = textread(fullfile(exp_home, 'imgnetchlg-wnid-1k.txt'), '%s');
%c2 = dir('/Users/xlx/Documents/proj/imgnet-flickr/wnet_tags/*txt')
%c2 = char(cat(1, c2.name));
%c2 = c2(:, 1:9);

[d2, c2] = textread(fullfile(exp_home, 'wnet.usrcnt.txt'), '%d%s');
c2 = char(c2);
c2 = c2(:, 3:11);

c3 = cellstr(c2(d2>0, :));
d3 = d2(d2>0);

[ctest, itest] = intersect(c3, c1);
[cother, iother] = setdiff(c3, c1);

[nt, xt] = hist(d3(itest), 50);

tmp = d3(iother) ;
dother = tmp(tmp>= min(d3(itest)));
cother = cother( tmp>= min(d3(itest)) ) ;
[no] = hist(dother, xt);

bar(xt, [nt(:), no(:)])

% xo = hist(d3(iother), 50);

%{
ii = randperm(length(cother));
fprintf(1, '%s\n', cother{ii(1:5)} );

ans = 

    'n07823814'
    'n09858299'
    'n04564278'
    'n10784113'
    'n13164501'
%}


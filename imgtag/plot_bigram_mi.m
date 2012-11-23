
clr = 'bgrcymk';

[synid, pb, mi, wordpair] = textread('/Users/xlx/proj/ImageNet/bg_birds_None2.txt', '%s%f%f%s%*[^\n]');
anm = unique(cellstr(synid));


figure; hold on;
for i=1:6, 
    ii=strmatch(anm{i}, synid); 
    plot(pb(ii), mi(ii), [clr(i) 'x'], 'markersize', 8);  
end
axis tight; grid on;
set(gca, 'xscale', 'log', 'yscale', 'log')
legend(anm, 'Location', 'SouthEast')
saveas(gcf, '/Users/xlx/proj/ImageNet/bg_birds_None2.png')

[synid, pb, mi, wordpair] = textread('/Users/xlx/proj/ImageNet/bg_birds_notNone2.txt', '%s%f%f%s%*[^\n]');
%anm = unique(cellstr(synid));

figure; hold on;
for i=1:6, 
    ii=strmatch(anm{i}, synid); 
    plot(pb(ii), mi(ii), [clr(i) 'x'], 'markersize', 8);  
end
axis tight; grid on;
set(gca, 'xscale', 'log', 'yscale', 'log')
legend(anm, 'Location', 'SouthEast')
saveas(gcf, '/Users/xlx/proj/ImageNet/bg_birds_notNone2.png')

% ----------- the bigger set / 50 synsets ----

[synid, pb, mi, wordpair] = textread('/Users/xlx/proj/ImageNet/bg_compiled_None.txt', '%s%f%f%s%*[^\n]');
anm = unique(cellstr(synid));


figure; hold on;
for i=1: length(anm), 
    ii=strmatch(anm{i}, synid); 
    plot(pb(ii), mi(ii), [clr(mod(i, length(clr))+1 ) 'x'], 'markersize', 8);  
end
axis tight; grid on;
set(gca, 'xscale', 'log', 'yscale', 'log')
legend(anm, 'Location', 'SouthEast')
saveas(gcf, '/Users/xlx/proj/ImageNet/bg_compiled_None.png')

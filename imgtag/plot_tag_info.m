

[~,hostn] = system('hostname');
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/db2';
end
out_mat = fullfile(data_dir, 'wordnet_tag_stat.mat');
fig_dir = fullfile(data_dir, 'fig');

load(out_mat, 'wvmat', 'vocab', 'wnlist', 'vcnt', 'vscore');

% plot tag frequency vs informativeness
% assume having vocab, vcnt, vscore 

nv = length(vocab);
x = 1 : nv;
[vs, iv] = sort(vcnt, 'descend');
vlabel = vocab(iv);
vfprct(iv) = 1 - (0 : nv-1)/nv ;
[~, jv] = sort(vscore);
vsprct(jv) = (1 : nv)/nv ;

%% 1st figure: frequency vs information
% recipe http://www.mathworks.com.au/help/techdoc/ref/plotyy.html
%[ax, h1, h2] = plotyy(x, vs, x, vscore(iv), 'loglog', 'semilogx');
[ax, h1, h2] = plotyy(x, vs, x, vscore(iv), 'semilogy', 'plot');

%[ax, h1, h2] = plotyy(x, log10(vs), x, vscore(iv), 'plot'); %, 'semilogx');

set(h1,'LineStyle','-') %, 'color', 'b')
set(h2,'LineStyle','x') %, 'color', 'g')

axes(ax(1)); axis tight
set(ax(1),'YTick',[10 100 1000 1e4], 'fontsize', 12)
set(ax(1),'XTick', [] )
set(get(ax(1),'Ylabel'),'String','tag frequency')

axes(ax(2)); grid on; axis tight
set(ax(2),'YTick', 0:0.5:3, 'fontsize', 12)
set(get(ax(2),'Ylabel'),'String','tag informativeness') 
set(get(ax(2),'XLabel'), 'String','tag rank by frequency' )

%xlabel('tag rank')
th = prctile(vscore, 98);
hold on;
plot([1; nv]*ones(1, length(th)), ones(2,1)*th, '-', 'color', [.4 .4 .4]);

saveas(gcf, fullfile(fig_dir,'tag_rank.pdf'));

%% second figure, rank-rank
figure(2); 
hr2 = plot(vfprct, vsprct, 'x', 'markersize', 4);
set(gca,'YTick', 0:0.2:1, 'fontsize', 12)
set(get(gca,'XLabel'), 'String','frequency percentile' )
set(get(gca,'YLabel'), 'String','information percentile' )
grid on; axis tight; 


make_rectangle_para = @(x) ( [x(1) x(3) x(2)-x(1) x(4)-x(3)] );
linclr2 = [.6 .3 .6]; % positive
linclr1 = [.3 .6 .3]; % negative informativeness
% zoom in, make text dots
figure; 
subplot(221);
txtrg1 = [.35 .5 .75 .92];
rctng1 = make_rectangle_para(txtrg1);
i1 = find(vfprct>=txtrg1(1) & vfprct<txtrg1(2) & vsprct>txtrg1(3) & vsprct<=txtrg1(4));
plot(vfprct(i1), vsprct(i1), '.', 'markersize', 3);
text(vfprct(i1), vsprct(i1), vocab(i1), 'color', linclr1, 'fontsize', 9);
axis(txtrg1);
grid on;
set(gca,'XTick', txtrg1(1):.1:txtrg1(2))
set(gca,'YTick', txtrg1(3):.1:txtrg1(4))
title('box A')
%set(get(gca,'XLabel'), 'String','frequency percentile' )
%set(get(gca,'YLabel'), 'String','information percentile' )

%figure; 
subplot(223);
txtrg2 = [.5 .75 .1 .3];
rctng2 = make_rectangle_para(txtrg2);
i2 = find(vfprct>=txtrg2(1) & vfprct<txtrg2(2) & vsprct>txtrg2(3) & vsprct<=txtrg2(4));
plot(vfprct(i2), vsprct(i2), '.', 'markersize', 3);
text(vfprct(i2), vsprct(i2), vocab(i2), 'color', linclr2, 'fontsize', 9);
axis(txtrg2);
grid on;
set(gca,'XTick', txtrg2(1):.1:txtrg2(2))
set(gca,'YTick', txtrg2(3):.1:txtrg2(4))
%set(get(gca,'XLabel'), 'String','frequency percentile' )
%set(get(gca,'YLabel'), 'String','information percentile' )
title('box C')

%figure;
subplot(222);
txtrg3 = [.85 .92 .92 1];
rctng3 = make_rectangle_para(txtrg3);
i3 = find(vfprct>=txtrg3(1) & vfprct<txtrg3(2) & vsprct>txtrg3(3) & vsprct<=txtrg3(4));
plot(vfprct(i3), vsprct(i3), '.', 'markersize', 3);
text(vfprct(i3), vsprct(i3), vocab(i3), 'color', linclr1, 'fontsize', 9);
axis(txtrg3);
grid on;
set(gca,'XTick', txtrg3(1):.02:txtrg3(2))
set(gca,'YTick', txtrg3(3):.02:txtrg3(4))
title('box B')

%figure;
subplot(224);
txtrg4 = [.8 .9 .4 .55];
rctng4 = make_rectangle_para(txtrg4);
i4 = find(vfprct>=txtrg4(1) & vfprct<txtrg4(2) & vsprct>txtrg4(3) & vsprct<=txtrg4(4));
plot(vfprct(i4), vsprct(i4), '.', 'markersize', 3);
text(vfprct(i4), vsprct(i4), vocab(i4), 'color', linclr2, 'fontsize', 9);
axis(txtrg4);
grid on;
set(gca,'XTick', txtrg4(1):.02:txtrg4(2))
set(gca,'YTick', txtrg4(3):.05:txtrg4(4))
title('box D')

saveas(gcf, fullfile(fig_dir,'tag_zoom.pdf'));

%% now draw boxes of magnification
figure(2); 
hold on; 

r1 = rectangle('Position', rctng1, 'edgecolor', linclr1, 'LineWidth', 2);
r2 = rectangle('Position', rctng2, 'edgecolor', linclr2, 'LineWidth', 2);
r3 = rectangle('Position', rctng3, 'edgecolor', linclr1, 'LineWidth', 2);
r4 = rectangle('Position', rctng4, 'edgecolor', linclr2, 'LineWidth', 2);

saveas(gcf, fullfile(fig_dir,'tag_scatter.pdf'));

%% draw all tags for exploration
figure(4);
ht_all = text(vfprct, vsprct, vocab, 'color', 'b', 'fontsize', 8);
grid on; axis([0 1 0 1]); 

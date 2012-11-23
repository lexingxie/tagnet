
%% script to generate plots in the supplemental material

cd /Users/xlx/Dropbox/Lexing_Docs/paper-draft/tagnet/supp-material/fig
dfile = '../matchbox_exp.xls' ;
% this will get all the data in mac
p_all = xlsread(dfile, 'A35:N44');

tw_cols = @(x) reshape(x, [], 2);
three_cols = @(x) reshape(x, [], 3);
co = colormap(jet(8));

%% data q = 150 
ii = find(p_all(:, 1) == 150);
nk = [3 5 7];
%alphs = tw_cols( p_all(ii, 2) );
%ks = tw_cols( p_all(ii, 3) );
%% just plot
figure; 
% multi-label test
subplot(131); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 5) ;
    cur_knn = p_all(ii(jj), 6) ;
    if k==3
        cur_prior = p_all(ii(jj), 4) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
axis([1 2000 .2 .9]); grid on;
title('EX1. collaborative tagging, q=150')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
legend(lgd_str, 'location', 'best')

% multilabel top 5
subplot(132); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 10) ;
    cur_knn = p_all(ii(jj), 11) ;
    if k==3
        cur_prior = p_all(ii(jj), 7) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
axis([1 2000 .05 .8]); grid on;
title('EX2. multi-tag annotation, q=150')
xlabel('\alpha (regularization parameter)');
ylabel('Precision @5 ');
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
legend(lgd_str, 'location', 'best')

% new data ap
subplot(133); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 13) ;
    cur_knn = p_all(ii(jj), 14) ;
    if k==3
        cur_prior = p_all(ii(jj), 12) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
axis([1 2000 .0 .3]); grid on;
title('EX3. new picture annotation, q=150')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
legend(lgd_str, 'location', 'best')

%orient landscape
saveas(gcf, 'q150.fig');
saveSameSize(gcf, 'format', 'png', 'file', 'q150.png')

%% data q = 250 and 300 
ii = find(p_all(:, 1) == 250);
nk = [3 5 7];
%alphs = tw_cols( p_all(ii, 2) );
%ks = tw_cols( p_all(ii, 3) );
%% just plot
figure; 
% multi-label test
subplot(131); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 5) ;
    cur_knn = p_all(ii(jj), 6) ;
    if k==3
        cur_prior = p_all(ii(jj), 4) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
axis([5 10000 .2 .9]); grid on;
title('EX1. collaborative tagging, q=250')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
legend(lgd_str, 'location', 'best')

% multilabel top 5
subplot(132); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 10) ;
    cur_knn = p_all(ii(jj), 11) ;
    if k==3
        cur_prior = p_all(ii(jj), 7) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
axis([5 10000 .05 .8]); grid on;
title('EX2. multi-tag annotation, q=250')
xlabel('\alpha (regularization parameter)');
ylabel('Precision @5 ');
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
legend(lgd_str, 'location', 'best')

% new data ap
subplot(133); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 13) ;
    cur_knn = p_all(ii(jj), 14) ;
    if k==3
        cur_prior = p_all(ii(jj), 12) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
axis([5 10000 .0 .3]); grid on;
title('EX3. new picture annotation, q=250')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
legend(lgd_str, 'location', 'best')

%orient landscape
saveas(gcf, 'q250.fig');
saveSameSize(gcf, 'format', 'png', 'file', 'q250.png')

%% data q = 250 and 300 
ii = find(p_all(:, 1) == 300);
nk = [3 5 7];
%alphs = tw_cols( p_all(ii, 2) );
%ks = tw_cols( p_all(ii, 3) );
%% just plot
figure; 
% multi-label test
subplot(131); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 5) ;
    cur_knn = p_all(ii(jj), 6) ;
    if k==3
        cur_prior = p_all(ii(jj), 4) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
axis([5 2000 .2 .9]); grid on;
title('EX1. collaborative tagging, q=300')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
legend(lgd_str, 'location', 'best')

% multilabel top 5
subplot(132); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 10) ;
    cur_knn = p_all(ii(jj), 11) ;
    if k==3
        cur_prior = p_all(ii(jj), 7) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
axis([5 2000 .05 .8]); grid on;
title('EX2. multi-tag annotation, q=300')
xlabel('\alpha (regularization parameter)');
ylabel('Precision @5 ');
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
legend(lgd_str, 'location', 'best')

% new data ap
subplot(133); 
lcnt = 0;
for k = nk
    jj = (p_all(ii, 3)==k) ;
    alph = p_all(ii(jj), 2);
    cur_mb =  p_all(ii(jj), 13) ;
    cur_knn = p_all(ii(jj), 14) ;
    if k==3
        cur_prior = p_all(ii(jj), 12) ;
        lcnt = lcnt + 1;
        semilogx(alph(:,1), cur_prior(:,1), '-', 'linewidth', 2, 'color', co(lcnt, :));
        hold on;
    end
    lcnt = lcnt + 1;
    semilogx(alph, cur_mb, '-*', 'linewidth', 2, 'color', co(lcnt, :))
    lcnt = lcnt + 1;
    semilogx(alph, cur_knn, '-x', 'linewidth', 2, 'color', co(lcnt, :))
end
axis([5 2000 .0 .3]); grid on;
title('EX3. new picture annotation, q=300')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
lgd_str = {'prior', 'matchbox K=3', 'knn K=3', 'matchbox K=5', 'knn K=5', 'matchbox K=7', 'knn K=7'};
legend(lgd_str, 'location', 'best')

%orient landscape
saveas(gcf, 'q300.fig');
saveSameSize(gcf, 'format', 'png', 'file', 'q300.png')

%% plot q = 400
ii = p_all(:, 1) == 400;
alphs = tw_cols( p_all(ii, 2) );
ks = tw_cols( p_all(ii, 3) );
%% just plot
figure; 
% multi-label test
subplot(131); %set(gca, 'colororder', co, 'nextplot', 'add');
cur_prior = tw_cols( p_all(ii, 4) );
cur_mb = tw_cols( p_all(ii, 5) );
cur_knn = tw_cols( p_all(ii, 6) );
semilogx(alphs(:,1), cur_prior(:,1), 'm-', 'linewidth', 2);
hold on;
semilogx(alphs, cur_mb, '-*', 'linewidth', 2)
semilogx(alphs, cur_knn, '-x', 'linewidth', 2)
axis([.5 20000 .2 .9]); grid on;
title('EX1. collaborative tagging, q=400')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
legend({'prior', 'matchbox K=3', 'matchbox K=5', 'knn K=3', 'knn K=5'}, 'location', 'best')
% multi-label test
subplot(132); %set(gca, 'colororder', co, 'nextplot', 'add');
cur_prior = tw_cols( p_all(ii, 7) );
cur_mb = tw_cols( p_all(ii, 10) );
cur_knn = tw_cols( p_all(ii, 11) );
semilogx(alphs(:,1), cur_prior(:,1), 'm-', 'linewidth', 2);
hold on;
semilogx(alphs, cur_mb, '-*', 'linewidth', 2)
semilogx(alphs, cur_knn, '-x', 'linewidth', 2)
axis([.5 20000 .05 .8]); grid on;
title('EX2. multi-tag annotation, q=400')
xlabel('\alpha (regularization parameter)');
ylabel('Precision @5 ');
legend({'prior', 'matchbox K=3', 'matchbox K=5', 'knn K=3', 'knn K=5'}, 'location', 'best')

% multi-label test
subplot(133); %set(gca, 'colororder', co, 'nextplot', 'add');
cur_prior = tw_cols( p_all(ii, 12) );
cur_mb = tw_cols( p_all(ii, 13) );
cur_knn = tw_cols( p_all(ii, 14) );
semilogx(alphs(:,1), cur_prior(:,1), 'm-', 'linewidth', 2);
hold on;
semilogx(alphs, cur_mb, '-*', 'linewidth', 2)
semilogx(alphs, cur_knn, '-x', 'linewidth', 2)
axis([.5 20000 .0 .3]); grid on;
title('EX3. new picture annotation, q=400')
xlabel('\alpha (regularization parameter)');
ylabel('AP on r_{ij}');
legend({'prior', 'matchbox K=3', 'matchbox K=5', 'knn K=3', 'knn K=5'}, 'location', 'best')

set(gcf, 'PaperPositionMode', 'auto');
%orient landscape
saveas(gcf, 'q400.fig');
saveSameSize(gcf, 'format', 'png', 'file', 'q400.png')


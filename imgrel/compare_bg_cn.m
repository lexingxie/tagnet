

[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
else % mac os
    data_dir = '/Users/xlx/Documents/proj/imgnet-flickr/db2';
end


%% load bigram
bg_mat = fullfile(data_dir, 'bg_flickr.mat');

if exist(bg_mat, 'file')
    load(bg_mat, 'bgcnt', 'cnprk', 'w1', 'w2'); 
else
    cn5_pr_mat = fullfile(data_dir, 'conceptnet-graph/CN5_pr.mat');
    load( cn5_pr_mat, 'G5p' );
    G5p = G5p + G5p';
    % Name               Size                  Bytes  Class             Attributes
    % G5              7797x7797               792592  double            sparse
    % G5p             7797x7797            486345672  double
    % alph               1x1                       8  double
    % word_idmap      7797x1                     112  containers.Map

    bg_file = fullfile(data_dir, 'flickr.bigram.ge5.txt' ) ;
    
    [bgcnt, w1, w2] = textread(bg_file, '%d%s%s');
    cnprk = -ones(size(bgcnt));
    for i = 1 : length(w1)
        if isKey(word_idmap, w1{i}) && isKey(word_idmap, w2{i})
            cnprk(i) = G5p(word_idmap(w1{i}), word_idmap(w2{i}) ); 
        end
    end
    bgcnt = bgcnt(cnprk>0);
    w1 = w1(cnprk>0);
    w2 = w2(cnprk>0);
    cnprk = cnprk(cnprk>0);
    
    save(bg_mat, 'bgcnt', 'cnprk', 'w1', 'w2');
end


ii = find(bgcnt>prctile(bgcnt,99) | cnprk>prctile(cnprk,99) );

x = log10(bgcnt)/log10(max(bgcnt) + eps('single')) ;
y =      (cnprk)/ (max(cnprk)+ eps('single')) ;
r = sqrt(x.^2 + y.^2) ;
theta = 180*atan(y./x)/pi ;

figure(1); semilogx(bgcnt, cnprk, '.'); axis tight; grid on;
figure(2); plot(x, y, 'g.'), axis([0 1 0 1]); grid on;
figure(3); plot(theta, r, 'r.'); grid on;

nump = 20;
% center region
jcent = find(theta>37.5 & theta<52.5 & r>0.35) ;
[~, ji] = sort(r(jcent), 'descend') ;
for i = 1 : nump
    ii = jcent(ji(i)) ;
    fprintf(1, '%0.4f \t %s \t %s \n', r(ii), w1{ii}, w2{ii} );
end
% bigram-prominent
jbg = find(theta<15 & r>0.35) ;
[~, ji] = sort(r(jbg), 'descend') ;
for i = 1 : nump
    ii = jbg(ji(i)) ;
    fprintf(1, '%0.4f \t %s \t %s \n', r(ii), w1{ii}, w2{ii} );
end

% concept-prominent
jcn = find(theta>68 & r>0.35) ;
[~, ji] = sort(r(jcn), 'descend') ;
for i = 1 : nump
    ii = jcn(ji(i)) ;
    fprintf(1, '%0.4f \t %s \t %s \n', r(ii), w1{ii}, w2{ii} );
end

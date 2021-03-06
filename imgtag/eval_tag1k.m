
exp_envsetup

result_mat_name = 'eval_Testall_20121123T215757.mat';
load(fullfile(data_dir, 'run-data', result_mat_name), 'U', 'V', 'Xtest', 'img_idx', 'Y');

eval_str = 'eval_tag1k_' ;


exp_setparams

%Rtest = Xtest'*U'*V*Y;

%% load tag features, setup Y for tag1k

load(fullfile(data_dir, 'tag_wn_feature.mat'), 'vocab', 'vscore', 'found_wn', 'target_tags')
% whos
%   Name                 Size                 Bytes  Class             Attributes
% 
%   found_wn            81x1                     81  logical                     
%   synset_map       13288x1                    112  containers.Map              
%   tag_feat            63x7323              765088  double            sparse    
%   target_tags         81x1                   9972  cell                        
%   vcnt              7323x1                  58584  double                      
%   vocab             7323x1                 915782  cell                        
%   vscore               1x7323               58584  double                      
%   wn_set              81x1                  19914  cell                        
%   wnlist           13288x1                1727440  cell                        
%   wvmat            13288x7323            57305376  double   sparse   

load( fullfile(data_dir, 'nuswide_1k_tagfeat.mat'), 'WG', 'col_label', 'row_label_WG')
%   Name                  Size                Bytes  Class     Attributes
% 
%   BG                17034x883            26815168  double    sparse  %bigram:tag-tag 
%   WG                19975x883            23857024  double    sparse  %wnet-tag
%   col_label           883x1                109386  cell                
%   row_label         20366x1               2577688  cell                
%   row_label_BG      17034x1               2151226  cell
%   row_label_WG      19975x1               2528202  cell      

load(fullfile(data_dir, 'TrainTest_Label.mat'), 'tag1k', 'tag81', 'train_tag_1k', 'test_tag_1k') ;
% >> whos('-file', fullfile(data_dir, 'TrainTest_Label.mat'))                                 
% Name                    Size                 Bytes  Class             Attributes
% 
%   tag1k                1000x1                 124242  cell                        
%   tag81                  81x1                   9972  cell                        
%   te_tag81           107859x81               1511696  double            sparse    
%   test_idx_map             -                     112  containers.Map              
%   test_label         107859x81               3238880  double            sparse    
%   test_tag_1k        107859x1000             9995816  double            sparse    
%   tr_tag81           161789x81               2249456  double            sparse    
%   train_idx_map            -                     112  containers.Map              
%   train_label        161789x81               4824000  double            sparse    
%   train_tag_1k       161789x1000            14971624  double
%   sparse   


load(fullfile(data_dir, 'Test_feat_objbank.mat'), 'imgid');
img_idx = cell2mat(values(test_idx_map, imgid));
test_img_name = imgid';
clear imgid

ntag = length(col_label); 

[rvocab, iv, iw]  = intersect(vocab, row_label_WG);
%tag1k_feat = WG(:, iw)';

[vs, jv] = sort(vscore, 'descend'); % take ~150 dimensions for now
% this was how training tag feature were taken
% tag_feat = tag_feat(:, iv(1: NUMV));
tag1k_feat = zeros(ntag, NUMV);
for i = 1 : NUMV
    if any(jv(i)==iv) % this is in row_labe_WG
        tag1k_feat(:, i) = WG(iw(jv(i)==iv), : )';
    else
        fprintf(1, 'not found: %s\n', vocab{jv(i)} );
    end
end

tag_known = target_tags(found_wn);

% find top tags
%[cvocab, j1, j2] = intersect(tag1k, col_label);
tagcnt = full(sum(train_tag_1k, 1));
[tagcnt, js] = sort(tagcnt, 'descend');

for i=1:81, tag81p{i}=[tag81{i}, 's']; end
for i=1:81, tag81s{i}=[tag81{i}(1:end-1)]; end
tag1k_reduced = setdiff(tag1k, [tag81; tag81p'; tag81s']);
clear tag81p tag81s

%Xtest = imgfeat'; this is loaded from model file
Y1k = log(tag1k_feat + 1)';
test_tag_col = test_tag_1k(img_idx, :);

nm = size(Yorig, 2);
knbr = 3 ; %[1 3 5 10];
tag_avg_dist = zeros(ntag, length(knbr));
tag_near_map = containers.Map('init', {'',1});
% ta = intersect(tag1k_reduced, tag1k(js(1:50)) );
for i = 1 : ntag
    dx = Y1k(:,i)*ones(1, nm) - Yorig;
    dy = sum(dx.^2, 1) ;
    [sd, idy] = sort(dy);
    tag_near_map(col_label{i}) = tag_known(idy(1:knbr(1)) );
    for j = 1 : length(knbr)
        tag_avg_dist(i, j) = sum(sd(1:knbr(j)))/knbr(j) ;
    end
end

[~, ia] = sort(tag_avg_dist, 1);
col_label(ia(1:10, :))
%col_label(ia(1:10,4))'

p10 = zeros(1, ntag);
tcnt = 0;
for i = 1 : 150 % 1 : 10 %length(tag1k)
    cur_tag = tag1k{js(i)};
    tt = strmatch(cur_tag, tag1k_reduced, 'exact');
    tj = strmatch(cur_tag, col_label, 'exact');
    if isempty(tt) || isempty(tj)
        continue;
    else
        tcnt = tcnt + 1;
    end
    Ri = Xtest' * U' * V * Y1k(:, tj);    
    curlab = test_tag_col(:, js(i));
    p_cur = compute_perf(Ri, 1.*full(curlab), 'store_raw_pr', 2, 'precision_depth', 10);
    p10(tj) = p_cur.p_at_d;
    % print score and filename for the top 10
    %fprintf(1, 'tag#%d "%s": \t p@10=%0.4f \t ap=%0.4f \t auc=%0.4f\n', tcnt, tag1k{js(i)}, p10(tj), p_cur.ap, p_cur.auc);
    fprintf(1, '# %0.4f\t %0.4f\t %0.4f\t %s\n', p_cur.p_at_d, p_cur.ap, p_cur.auc, cur_tag);
    if 0
        [~, rj] = sort(Ri, 'descend');
        %disp ( test_img_name(rj(1:10)) )
        % cp $NUSIMGDIR/food/0531_29900246.jpg test/01_food_0531_29900246.jpg
        fprintf(1, '\nmkdir %s\n', cur_tag);
        for j = 1 : 10
            src_name = test_img_name{rj(j)} ;
            fprintf(1, 'cp $NUS_IMG_DIR/%s %s/%02d_%s\n', src_name, tag1k{js(i)}, j, strrep(src_name, '/', '_') );
        end
        disp(' ');
    end
end

clear Xtest


save(sav_file, 'Xtest', 'U', 'V', 'Y1k', 'col_label', 'tag1k', 'tag81', 'test_tag_col', 'test_img_name') 
diary off

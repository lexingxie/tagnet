

load(fullfile(data_dir, 'run-data', result_mat_name), 'U', 'V', 'Xtest', 'img_idx');

eval_str = 'eval_tag1k_' ;

exp_envsetup
exp_setparams

Rtest = Xtest'*U'*V*Y;

%% load tag features, setup Y for tag1k

load(fullfile(data_dir, 'tag_wn_feature.mat'), 'vocab', 'vscore')
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

load(fullfile(data_dir, 'TrainTest_Label.mat'), 'test_tag_1k') ;
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

ntag = length(col_label); 

[rvocab, iv, iw]  = intersect(vocab, row_label_WG);
%tag1k_feat = WG(:, iw)';

[vs, jv] = sort(vscore, 'descend'); % take ~150 dimensions for now
% this was how training tag feature were taken
% tag_feat = tag_feat(:, iv(1: NUMV));
tag1k_feat = zeros(ntag, NUMV);
for i = 1 : NUMV
    if any(jv(i)==iv) % this is in row_labe_WG
        tag1k_feat(:, i) = WG(:, iw(jv(i)==iv) );
    else
        fprintf(1, 'not found: %s\n', vocab(jv(i)) );
    end
end


% find top tags
%[cvocab, j1, j2] = intersect(tag1k, col_label);
tagcnt = sum(train_tag_1k, 2);
[tagcnt, js] = sort(tagcnt, 'descend');

%Xtest = imgfeat'; this is loaded from model file
Y1k = log(tag1k_feat + 1)';

p10 = zeros(1, ntag);
for i = 1 : 10 %length(tag1k)
    Ri = Xtest' * U' * V * Y1k(:, i);
    tj = strmatch(tag1k{js(i)}, col_label, 'exact');
    curlab = test_tag_1k(img_idx, js(i));
    p_cur = compute_perf(Ri, 1.*full(curlab), 'store_raw_pr', 2, 'precision_depth', 10);
    p10(tj) = p_cur.p_at_d;
    % print score and filename for the top 10
    fprintf(1, 'tag %s: p@10=%0.4f, ap=%0.4f, top 10 images\n', tag1k{js(i)}, p10(tj), p_cur.ap);
    
end

clear Xtest

save(sav_file) 
diary off

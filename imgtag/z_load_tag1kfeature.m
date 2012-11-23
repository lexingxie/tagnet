data_dir = '/Users/xlx/Documents/proj/imgnet-flickr/nuswide' ;
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

load(fullfile(data_dir, 'TrainTest_Label.mat'), 'train_tag_1k') ;
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

[rvocab, iv, iw]  = intersect(vocab, row_label_WG);
tag1k_feat = WG(:, iw)';
[~, jv] = sort(vscore(iv), 'descend');

[cvocab, j1, j2] = intersect(tag1k, col_label);
tagcnt = sum(train_tag_1k, 2);
[tagcnt, js] = sort(tagcnt, 'descend');

save(fullfile(data_dir, 'tag1k_cooc_feature.mat'), 'vocab', 'vscore')
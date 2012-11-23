%% -----------
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/nuswide2';
    db_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/nuswide2';
    db_dir = '/Users/xlx/proj/ImageNet/db2';
end


%% prep train/test labels

out_mat_name = 'NUSwide_train_test_label.mat';

if exist(fullfile(data_dir, out_mat_name), 'file')
    load(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', '*tag*', 'tag_co');
else
    
    
    % map tags
    sql_dict = fullfile(db_dir, 'dict.db');
    mksqlite('open', sql_dict);
    res = mksqlite('SELECT string,word FROM dict');
    mksqlite('close');
    str_map = containers.Map({res.string}, {res.word});
    addl_words = textread(fullfile(db_dir, 'places_etc.txt'), '%s');
    str_map = [containers.Map(addl_words, addl_words); str_map];
    
    in_tag1k = textread(fullfile(data_dir, 'TagList1k.txt'), '%s');
    in_tag5k = textread(fullfile(data_dir, 'Final_Tag_List.txt'), '%s');
    
    flag_1k = isKey(str_map, in_tag1k) ;
    flag_5k = isKey(str_map, in_tag5k) ;
    out_tag1k = unique(values(str_map, in_tag1k(flag_1k)) );
    out_tag5k = unique(values(str_map, in_tag5k(flag_5k)) );
    
    % already loaded via load_TrainTest_label81.m
    load(fullfile(data_dir, 'TrainTest_Label.mat'), '*_tag_1k');
    
    num1 = length(out_tag1k);
    
    train_label_1k = false(size(train_tag_1k,1), num1);
    test_label_1k  = false(size(test_tag_1k ,1), num1);
    invidx = cell(num1, 1);
    for j = 1 : length(in_tag1k)
        if flag_1k(j)
            out_tag = str_map(in_tag1k{j});
            jj = strmatch(out_tag, out_tag1k, 'exact');
            invidx{jj} = [invidx{jj}, j];
        end
        
    end
    for i = 1 : num1
        train_label_1k(:, i) = sum(train_tag_1k(:, invidx{i}), 2)>0 ;
        test_label_1k(:, i)  = sum(test_tag_1k(:, invidx{i}), 2)>0  ;
    end
    
    
    %% compute coocurrence
    
    tag_co = zeros(num1, num1);
    for i = 1 : num1
        for j = 1 : i - 1
            tag_co(i, j) = sum(train_label_1k(i, :) & train_label_1k(j, :));
        end
    end
    tag_co = tag_co + tag_co' ;
    
    %% write results
    fo = fopen(fullfile(data_dir, 'vocab_nuswide1k.txt'), 'wt');
    fprintf(fo, '%s\n', out_tag1k{:});
    fclose(fo);
    fo = fopen(fullfile(data_dir, 'vocab_nuswide5k.txt'), 'wt');
    fprintf(fo, '%s\n', out_tag5k{:});
    fclose(fo);
    save(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', 'in_tag*', 'out_tag*', 'tag_co');
    
end
    

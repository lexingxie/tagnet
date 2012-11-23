
%% -----------
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/sun09';
else % mac os
    data_dir = '/Users/xlx/proj/ImageNet/sun09';
end

%% prep train/test labels
%data_dir = '/Users/xlx/proj/ImageNet/sun09';
mat_name = 'sun09_labels.mat';
out_mat_name = 'sun09_train_test.mat';
sun09_det_mat = 'sun09_detectorOutputs.mat';
if exist(fullfile(data_dir, out_mat_name), 'file')
    load(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', '*objname*');
else
    load(fullfile(data_dir, mat_name));
    
    % training data
    tmpanno = [Dtraining(:).annotation];
    
    train_imgname = {tmpanno.filename};
    n = length(train_imgname);
    
    tmpobj = {tmpanno.object};
    %tmpn = [tmpanno.object];
    tmpn = cell(n, 1);
    for k = 1 : n
        tmpn{k} = {tmpobj{k}.name};
    end
    objname = names; %% 111 labels %unique(cat(2, tmpn{:}));
    m = length(objname);
    train_label = false(n, m);
    for k = 1: n
        [~, jn] = intersect(objname, tmpn{k});
        train_label(k, jn) = true;
    end
    
    % test data
    tmpanno = [Dtest(:).annotation];
    
    test_imgname = {tmpanno.filename};
    n = length(test_imgname);
    
    tmpobj = {tmpanno.object};
    tmpn = cell(n, 1);
    for k = 1 : n
        tmpn{k} = {tmpobj{k}.name};
        
    end
    %objname = unique(cat(2, tmpn{:}));
    %m = length(objname);
    test_label = false(n, m);
    for k = 1: n
        [~, jn] = intersect(objname, tmpn{k});
        test_label(k, jn) = true;
    end
    
    %% map object names
    sql_dict = fullfile(data_dir, '../db2', 'dict.db');
    mksqlite('open', sql_dict);
    res = mksqlite('SELECT string,word FROM dict');
    mksqlite('close');
    str_map = containers.Map({res.string}, {res.word});
    
    objname_mapped = cell(length(objname), 1);
    for i = 1 : length(objname)
        if isKey(str_map, objname{i})
            objname_mapped{i} = str_map(objname{i});
        else
            objname_mapped{i} = '';
        end
    end
    objname_mapped{33} = 'counter';
    
    new_objname = unique(objname_mapped);
    numo = length(new_objname);
    train_label_new = false(size(train_label,1), numo);
    test_label_new  = false(size(test_label ,1), numo);
    
    for j = 1 : length(objname_mapped)
        jj = strmatch(objname_mapped{j}, new_objname, 'exact');
        tmp = train_label_new(:, jj) | train_label(:, j) ;
        train_label_new(:, jj) = tmp;
        tmp = test_label_new(:, jj) | test_label(:, j) ;
        test_label_new(:, jj) = tmp;
        
    end
    
    save(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', 'objname');
    
    
    %% read features
    feat_path = textread(fullfile(data_dir, 'feat_file_list.txt'), '%s');
    feat_list = textread(fullfile(data_dir, 'feat_imgnodir.txt'), '%s');
    
    feat_home = '~/vault-xlx/SUN09/objbank';
    
    NUM_MODELS = 177;
    
    % training features
    n = length(train_imgname);
    train_flag = true(n, 1);
    train_feat = zeros(n, NUM_MODELS);
    for k = 1: n
        ii = strmatch(train_imgname{k}, feat_list, 'exact');
        if length(ii)==1
            feat_file = fullfile(feat_home, feat_path{ii});
            cur_feat = load(feat_file);
            train_feat(k, :) = max(reshape(cur_feat, NUM_MODELS, []), [], 2);
        else
            train_flag(k) = false;
            fprintf(1, 'err matching entry %d/%d %s, ii = [%s]\n', k, n, train_imgname{k}, num2str(ii));
        end
    end
    fprintf(1, 'found %d of %d features in training set\n\n', sum(train_flag), n);
    %train_imgname = train_imgname(train_flag);
    train_feat = train_feat(train_flag, :);
    train_imgname = train_imgname(:);
    
    % test features
    n = length(test_imgname);
    test_flag = true(n, 1);
    test_feat = zeros(n, NUM_MODELS);
    for k = 1: n
        ii = strmatch(test_imgname{k}, feat_list, 'exact');
        if length(ii)==1
            feat_file = fullfile(feat_home, feat_path{ii});
            cur_feat = load(feat_file);
            test_feat(k, :) = max(reshape(cur_feat, NUM_MODELS, []), [], 2);
        else
            test_flag(k) = false;
            fprintf(1, 'err matching entry %d/%d %s, ii = [%s]\n', k, n, test_imgname{k}, num2str(ii));
        end
    end
    fprintf(1, 'found %d of %d features in training set\n\n', sum(test_flag), n);
    test_feat = test_feat(test_flag, :);
    test_imgname = test_imgname(:);
    
    save(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', '*objname*');
end
    
%% compute coocurrence on sun'09 data

nobj = length(new_objname);
obj_co = zeros(nobj, nobj);
for i = 1 : nobj
    for j = 1 : i - 1
        obj_co(i, j) = sum(train_label_new(i, :) & train_label_new(j, :));
    end
end
obj_co = obj_co + obj_co' ;
save(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', '*objname*', 'obj_co');


%% aggregate detector score for SUN09 input
topK = 3;
numo = length(new_objname);

load(fullfile(data_dir, sun09_det_mat), 'DdetectorTraining');

num_img = length(train_imgname);

train_detector = -ones(num_img, numo);
train_detector_max = -ones(num_img, numo);
for i = 1 : num_img
    objlist = {DdetectorTraining(i).annotation.object.name};
    conflist = [DdetectorTraining(i).annotation.object.confidence];
    olist = unique(objlist);
    for o = 1 : length(olist)
        oj = strcmp(olist{o}, objlist);
        oconf = sort(conflist(oj), 'descend');
        if length(oconf)>= topK
            oscore = mean(oconf(1: topK));
        else
            oscore = max(oconf);
        end
        jtmp = strcmp(olist{o}, objname);
        fj = strcmp(objname_mapped{jtmp}, new_objname);
        
        train_detector(i, fj) = oscore ;
        train_detector_max(i, fj) = max(oconf);
    end
    if mod(i, 1000)==0
        fprintf(1, ' %d traing imgs processed\n', i);
    end
end
clear DdetectorTraining

load(fullfile(data_dir, sun09_det_mat), 'DdetectorTest');
num_img = length(test_imgname);

test_detector = -ones(num_img, numo);
test_detector_max = -ones(num_img, numo);
for i = 1 : num_img
    objlist = {DdetectorTest(i).annotation.object.name};
    conflist = [DdetectorTest(i).annotation.object.confidence];
    olist = unique(objlist);
    for o = 1 : length(olist)
        oj = strcmp(olist{o}, objlist);
        oconf = sort(conflist(oj), 'descend');
        if length(oconf)>= topK
            oscore = mean(oconf(1: topK));
        else
            oscore = max(oconf);
        end
        jtmp = strcmp(olist{o}, objname);
        fj = strcmp(objname_mapped{jtmp}, new_objname);
        
        test_detector(i, fj) = oscore ;
        test_detector_max(i, fj) = max(oconf);
    end
    if mod(i, 1000)==0
        fprintf(1, ' %d test imgs processed\n', i);
    end
end

clear DdetectorTest ;
save(fullfile(data_dir, out_mat_name), 'train_*', 'test_*', '*objname*', 'obj_co');

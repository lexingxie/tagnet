

%% -----------
[~,hostn] = system('hostname');
hostn = deblank(hostn);
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves')
    db_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/db2';
else % mac os
    db_dir = '/Users/xlx/proj/ImageNet/db2';
end


wnet_tag_mat = fullfile(db_dir, 'wnet_tag_map.mat');
if ~exist(wnet_tag_mat, 'file')
    %% load dictionary
    sql_dict = fullfile(db_dir, 'dict.db');
    mksqlite('open', sql_dict);
    res = mksqlite('SELECT string,word,is_stopword FROM dict');
    mksqlite('close');
    str_map = containers.Map({res.string}, {res.word});
    stopword_map = containers.Map({res.string}, {res.is_stopword});
    
    addl_words = textread(fullfile(db_dir, 'places_etc.txt'), '%s');
    str_map = [containers.Map(addl_words, addl_words); str_map];
    stopword_map = [containers.Map(addl_words, num2cell(zeros(length(addl_words),1)) ); stopword_map];
    
    % load vocab
    tag_map = containers.Map();
    
    % load wordnet
    wnet_list = textread(fullfile(db_dir, 'wnet-50.txt'), '%*d%s');
    mksqlite('open', fullfile(db_dir, 'wordnet.db'));
    wnet_map = containers.Map();
    wnet_id_map = containers.Map(wnet_list, num2cell(1:length(wnet_list)));
    for i = 1 : length(wnet_list)
        wnid = wnet_list{i};
        res = mksqlite(sprintf('SELECT allwords FROM wordnet WHERE wnid=''%s''', wnid));
        tmp = textscan(res.allwords, '%s', 'delimiter', ' ,', 'MultipleDelimsAsOne', true);
        cur_words = lower(tmp{1}(~cellfun('isempty', tmp{1})) );
        cur_words = unique(cur_words);
        
        flag_w = isKey(str_map, cur_words) ;
        out_curw = values(str_map, cur_words(flag_w)) ;
        out_wflag = cell2mat(values(stopword_map, cur_words(flag_w)) ) ;
        out_curw = unique(out_curw(~out_wflag));
        
        wnet_map(wnid) = out_curw;
        for w = 1 : length(out_curw)
            if isKey(tag_map, out_curw{w})
                tag_map(out_curw{w}) = [wnid, tag_map(out_curw{w})] ;
            else
                tag_map(out_curw{w}) = {wnid};
            end
        end
        if mod(i, 500)==0
            fprintf(1, '%d / %d wnid mapped, %d tags found\n', i, length(wnet_list), double(tag_map.Count) );
        end
    end
    mksqlite('close');
    fprintf(1, '%d / %d wnid mapped, %d tags found\n', i, length(wnet_list), double(tag_map.Count) );
    save(wnet_tag_mat, 'wnet_map', 'wnet_id_map', 'tag_map');
else
    load(wnet_tag_mat, 'wnet_map', 'wnet_id_map', 'tag_map');
end




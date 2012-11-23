    

top_tag20 = textread(fullfile(data_dir, 'tags_top20.txt'));
cur_tag = top_tag20{k} ;

eval_str = sprintf('hidlabel_%02d_%s_', k, cur_tag) ;

sav_dir = fullfile(data_dir, 'run-data');
log_timestamp = datestr(now, 30);
sav_file = fullfile(sav_dir, [eval_str, log_timestamp '.mat']);
cur_diary = fullfile(sav_dir, [eval_str, log_timestamp '.diary']);

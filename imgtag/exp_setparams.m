
% set common parameters for tagnet experiments

%NUMI = [5000   20000   3000  1000]; %[100   200   300  1000]; 
% NUMI[1:2] # of imgs with 5+ tags, # with <=5 tags ==> training data
% NUMI[3:4]

% # of dimension for tag features
NUMV = 150 ;

% # of hidden dimensions
K = 5; %[3 5 7] ; %9:-2:5;
alph = 100 ; %[1 10 100 1000 10000]; 
max_iter = 5; %25

neg_pos_ratio = 8;
max_num_pos = 1e4; 
max_num_neg = 2e4;

if ~exist(log_timestamp, 'var')
    sav_dir = fullfile(data_dir, 'run-data');
    log_timestamp = datestr(now, 30);
    sav_file = fullfile(sav_dir, [eval_str, log_timestamp '.mat']);
    cur_diary = fullfile(sav_dir, [eval_str, log_timestamp '.diary']);
end

diary(cur_diary)

fprintf('logging to %s\n', cur_diary);
fprintf('settings:\n\tNUMV=%d\n', NUMV);
%fprintf('\tNUMI= %s\n', num2str(NUMI) );
fprintf('\tK= %s\n', num2str(K) );
fprintf('\talpha= %s\n', num2str(alph));
fprintf('\tmax_iter= %d\n', max_iter);
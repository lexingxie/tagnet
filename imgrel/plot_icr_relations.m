
% mperf = eval_conceptrank(CN_new, W, CN4, 'prec_depth', 10, 'eval_mode', 'macro', 'verbose', 0) ;
% bmperf = eval_conceptrank(CN_new, B, CN4, 'prec_depth', 10, 'eval_mode', 'macro', 'verbose', 0) ;
% mat_name = fullfile(exp_home, exp_subdir, sprintf('%s_n%03d.mat', log_name, syn_cnt) );

data_dir = '/Users/xlx/Documents/proj/imgnet-flickr'; 
exp_dir  = 'conceptrank-exp/ilsvrc-eval';
mat_file1 = 'ilsvrc_eval_20130309T105504_n050.mat';
mat_file2 = 'ilsvrc_eval_20130309T105504_n400.mat'; %'ilsvrc_eval_20130307T212711.mat';
mat_name1 = fullfile(data_dir, exp_dir, mat_file1); 
mat_name2 = fullfile(data_dir, exp_dir, mat_file2); 

load(mat_name1, 'all_t', 'mperf', 'W', 'CN*', 'B');
[mp1, W1] = deal(mperf, W);

load(mat_name2, 'mperf', 'W');
[mp2, W2] = deal(mperf, W);

clear mperf W
% mperf = 
%        idx: [1x1524 double]
%         ap: [1x1524 double]
%         f1: [1x1524 double]
%     p_at_d: [1x1524 double]
%      prior: [1x1524 double]
%       npos: [1x1524 double]

%print_top_pairs(triu(W, 1), all_t, 20, triu(CN4, 1), triu(CN5, 1), 'icr', mperf.idx(ii(2)) );
nw = length(mp1.idx);

w1 = all_t(mp1.idx);
pd = mp1.p_at_d'*[1 1] ;
npos = mp1.npos;

figure(1); hold on; 
cnt = [0 0 0]; % increase, decrese, no-change
print_cnt = 0;
for i = 1 : nw
    ii = find(mp2.idx == mp1.idx(i) );
    pd(i, 2) = mp2.p_at_d(ii);
    if pd(i, 2) > pd(i, 1)
        cnt(1) = cnt(1) + 1;
        semilogx(npos(i)*[1 1], pd(i,:), 'x-');
        
        if npos(i)>10 && npos(i)<=25 && pd(i,2)>=0.6 && pd(i,2)<=0.8 && diff(pd(i,:))>=.3
            fprintf(1, '\n\n %d\t %0.2f\t%0.2f\n', npos(i), pd(i,:) );
            print_icr_rel(B,  CN4, CN5, mp1.idx(i), all_t, 10, false) ;
            print_icr_rel(W1, CN4, CN5, mp1.idx(i), all_t, 10, false) ;
            print_icr_rel(W2, CN4, CN5, mp1.idx(i), all_t, 10, false) ;
            
            disp('');
            print_cnt = print_cnt + 1;
            %print_top_pairs(triu(W1, 1), all_t, 10, triu(CN4, 1), triu(CN5, 1), '', mp1.idx(i) );
            %print_top_pairs(triu(W2, 1), all_t, 10, triu(CN4, 1), triu(CN5, 1), '', mp1.idx(i) );            
        end
        
    elseif pd(i, 2) < pd(i, 1)
        cnt(2) = cnt(2) + 1;
    else
        cnt(3) = cnt(3) + 1;
    end
end
disp(cnt);
fprintf(1, 'print_cnt = %d\n', print_cnt);

axis tight; hold off; grid on;


ns = 25:25:462 ;
max_cn = 7796;

if 1
    data_dir = '/Users/xlx/Documents/proj/imgnet-flickr/conceptrank-exp'; 
    result_file = fullfile(data_dir, 'ilsvrc-eval/join_180_cleaned.txt');
    
    [max_re, ap_b, ap_w, num_cn] = textread(result_file, '%*s%f%*s%f%*s%f%d');
    
else % windows
    xls_dir = '/Users/xlx/Dropbox/ToXuming/pagerank-exp';
    xls_name = fullfile(xls_dir, 'matchbox-perf.mm13.xlsx') ;
    
    xls_sheet = 'ICR_stat';
    row_s = 1;
    row_e = 180;
    
    w_col = 'O';
    ap_w = xlsread(xls_name, xls_sheet, sprintf('%s%d:%s%d', w_col, row_s, w_col, row_e) );
    
    b_col = 'I';
    ap_b = xlsread(xls_name, xls_sheet, sprintf('%s%d:%s%d', b_col, row_s, b_col, row_e) );
    
    
    mr_col = 'D';
    max_re = xlsread(xls_name, xls_sheet, sprintf('%s%d:%s%d', mr_col, row_s, mr_col, row_e) );
    
    
    cn_col = 'D';
    num_cn = xlsread(xls_name, xls_sheet, sprintf('%s%d:%s%d', cn_col, row_s, cn_col, row_e) );
    
end
colstat = @(x) [mean(reshape(x, [], 10), 2), std(reshape(x, [], 10), 0, 2) ]' ;

max_res = colstat(max_re);
ap_bs = colstat(ap_b);
ap_ws = colstat(ap_w);
num_cns = colstat(num_cn)/max_cn;


figure(1);  errorbar(ns, ap_ws(1, :), ap_ws(2, :), 'r--');       hold on;
            plot([ns(1); ns(end)]*[1 1], [1;1]*[min(ap_b), max(ap_b)], '-.');    
            axis([0 inf 0 inf]);    hold off;    grid on;
figure(2);  errorbar(ns, num_cns(1, :), num_cns(2, :), '--');   hold on;
            errorbar(ns, max_res(1, :), max_res(2, :), 'r--');  grid on; axis tight;
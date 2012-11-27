
function query_top10img(cur_tag, Xtest, U, V, Y1k, col_label, tag1k, tag81, test_tag_col, test_img_name)


for i=1:81, tag81p{i}=[tag81{i}, 's']; end
for i=1:81, tag81s{i}=[tag81{i}(1:end-1)]; end
tag1k_reduced = setdiff(tag1k, [tag81; tag81p'; tag81s']);

%cur_tag = tag1k{js(i)};
tt = strmatch(cur_tag, tag1k_reduced, 'exact');
tj = strmatch(cur_tag, col_label, 'exact');
ti = strmatch(cur_tag, tag1k, 'exact');

if isempty(tt) || isempty(tj)
    fprintf(1, '# tag %s not found\n', cur_tag);
    return; 
end

Ri = Xtest' * U' * V * Y1k(:, tj);
%curlab = test_tag_1k(img_idx, ti);
curlab = test_tag_col(:, ti);

p_cur = compute_perf(Ri, 1.*full(curlab), 'store_raw_pr', 2, 'precision_depth', 10);

% print score and filename for the top 10
%fprintf(1, 'tag#%d "%s": \t p@10=%0.4f \t ap=%0.4f \t auc=%0.4f\n', tcnt, tag1k{js(i)}, p10(tj), p_cur.ap, p_cur.auc);
fprintf(1, '# %0.4f\t %0.4f\t %0.4f\t %s\n', p_cur.p_at_d, p_cur.ap, p_cur.auc, cur_tag);

[~, rj] = sort(Ri, 'descend');

%disp ( test_img_name(rj(1:10)) )
% cp $NUSIMGDIR/food/0531_29900246.jpg test/01_food_0531_29900246.jpg
fprintf(1, '\nmkdir %s\n', cur_tag);
for j = 1 : 10
    src_name = test_img_name{rj(j)} ;
    fprintf(1, 'cp $NUS_IMG_DIR/%s %s/%02d_%s\n', src_name, cur_tag, j, strrep(src_name, '/', '_') );
end
disp(' ');



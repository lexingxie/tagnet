
ta = intersect(tag1k_reduced, tag1k(js(1:50)) );

for i = 1 : length(ta)
    query_top10img(ta{i}, Xtest, U, V, Y1k, col_label, tag1k, tag81, test_tag_col, test_img_name);
end

for i=1:length(ta), 

    if isKey(tag_near_map, ta{i})
        wl = tag_near_map(ta{i});
        fprintf(1, '# %s :\t\t ',  ta{i});
        fprintf(1, '%s ', wl{:});
        fprintf(1, '\n');
    end
end
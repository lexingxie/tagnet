

% batch covert 1-channel jpg to 3 channels

root_dir = '~/vault-xlx/SUN09/Images/static_sun09_database';

dd = dir(root_dir);

for d  = 1 : length(dd)
    dn = dd(d).name;
    if strcmp(unique(dn), '.')
        continue;
    end
    if dd(d).isdir 
        fprintf(1, 'processing dir "%s" ... \n', dn);
        ss = dir(fullfile(root_dir, dn));
        for s = 1 : length(ss)
            if length(ss(s).name)>=4 && strcmp(ss(s).name(end-3:end), '.jpg')
                onechan_2_threechan(fullfile(root_dir, dn, ss(s).name))
            end
        end
    elseif strcmp(dd(d).name(end-3:end), '.jpg')
        onechan_2_threechan(fullfile(root_dir, dd(d).name))
    else
        %skip
    end
end
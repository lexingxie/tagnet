function onechan_2_threechan(filename)

im = imread(filename);
if size(im, 3) == 1
    img = repmat(im, [1 1 3]);
    [p, n, e] = fileparts(filename);
    if ~strcmp(e, '.jpg')
       filename = fullfile(p, [n, '.jpg']);
    end
    imwrite(img, filename, 'quality', 90);
    fprintf(1, ' converted %s\n', filename);
end
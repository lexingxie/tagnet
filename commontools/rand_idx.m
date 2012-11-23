

function idx = rand_idx(N, m)
% return m random ids (or items) from a collection of N items
% wrap around randperm

if numel(N)==1
    tmp = randperm(N);
    m = min(N, m);
    idx = tmp(1:m);
else
    tmp = randperm(numel(N));
    m = min(m, length(tmp));
    idx = N(tmp(1:m));
end
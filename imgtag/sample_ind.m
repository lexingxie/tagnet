function ind = sample_ind(in_label_mat, neg_ratio, num_fold)
% sample a 0/1 matrix: index manipulation
% specify the ratio of 0s to 1s (assume keeping all 1s)
% specify the number of folds it need
%
% return a vector of linear indexes, or NUM_FOLD cells each containing
% disjoint indexes

if nargin<3
    num_fold = 1;
end

ipos = find(in_label_mat==1);
ineg = find(in_label_mat==0);

npos = length(ipos);
nneg = length(ineg);

num_neg = min([nneg, round(neg_ratio*npos)]);

jneg = rand_idx(ineg, num_neg);

if num_fold == 1
    ind = [ipos; jneg];
else
    ifp = randperm(npos);
    ifn = randperm(num_neg);
    stpp = floor(npos/num_fold);
    stpn = floor(num_neg / num_fold);
    
    ind = cell(num_fold, 1);
    for f = 1 : num_fold
        ind{f} = [ipos(ifp((f-1)*stpp+1: f*stpp));
            jneg(ifn((f-1)*stpn+1: f*stpn))] ;
    end
end

fprintf(1, '%s returning %d pos-id and %d neg-id from %d x %d matrix\n', ...
    datestr(now, 31), npos, num_neg, size(in_label_mat) );
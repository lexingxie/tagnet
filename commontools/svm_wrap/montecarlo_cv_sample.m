function [out_ind, varargout] = montecarlo_cv_sample(orig_ind, sample_portion, labels, varargin)
% input: 
%   orig_ind -- a list of data indexes
%   sample_portion -- percentage of the random sample
%   labels -- if available, sample proportional w.r.t. each label
%
% 2006-04-02, xlx@us.ibm.com

[seed] = process_options(varargin, 'seed', 0);

assert(all(sample_portion<=1), 'sample percentage need to be <1')

if ~exist('labels', 'var') || isempty(labels)
    is_exclusive = length(sample_portion>1);
    sample_amount = round(length(orig_ind)*sample_portion + rand/2.01); 
    % add rand to avoid always under-sampling for small sample_portion
    sample_ind = take_sample_ind(sample_amount, length(orig_ind), is_exclusive);
    if length(sample_ind) ==1
        out_ind = orig_ind(sample_ind{1});
        remainder_ind = setdiff(orig_ind, out_ind);        
    else
        for i = 1 : length(sample_ind)
            out_ind{i} = orig_ind(sample_ind{i});
        end
        remainder_ind = cell(1, length(sample_ind));
    end
else
    all_lab = unique(labels);
    [out_ind, remainder_ind] = deal(cell(length(all_lab), length(sample_portion)));
    for j = 1 : length(all_lab)
        cur_id = orig_ind(labels==all_lab(j));
        if length(sample_portion)>1
            [out_ind(j, :), remainder_ind(j,:)] = montecarlo_cv_sample(cur_id, sample_portion);
        else
            [out_ind{j}, remainder_ind{j}] = montecarlo_cv_sample(cur_id, sample_portion);
        end
    end
    if length(sample_portion) == 1
        if sum(cellfun('size', out_ind,1))>sum(cellfun('size', out_ind,2)), dim = 1; else, dim = 2; end
        out_ind = cat(dim, out_ind{:}); 
        remainder_ind = cat(dim, remainder_ind{:});
    end
end

varargout = {remainder_ind};

% -----------------------------
function sample_ind = take_sample_ind(sample_amount, total_num, is_exclusive)

baseperm = randperm(total_num);
sample_ind = cell(1, length(sample_amount));
for i = 1 : length(sample_amount)
    sample_ind{i} = sort(baseperm(1 : sample_amount(i)));
    if ~is_exclusive
        baseperm = randperm(total_num_ex);
    else
        baseperm = baseperm([sample_amount(i)+1:end, 1:sample_amount(i)]);
        % move the things already taken to the end, not to be touched again
    end
end
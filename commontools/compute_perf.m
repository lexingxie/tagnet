function perf = compute_perf(scores, labels, varargin)
% perf = compute_perf(scores, labels, varargin)
% Rank-list evaluation script
% given vector (or vectors) of scores
% compute an assortment of performance metrics 
% precision, recall, equal-precision-recall, average precision
%
% 2004-12-13, xlx
% 2006-04-11, revised for all flavors of metrics

[ap_depth, ~, ap_range, pos_label, neg_label, pr_nsample, precision_depth, get_accuracy, ...
    re_thresh, randomize_first, store_raw_pr, verbose] ...
    = process_options(varargin, ...
    'ap_depth', -1, 'ap_no_ignore', true, 'ap_range', [], 'pos_label', 1, 'neg_label', 0, 'pr_nsample', -1, ...
    'precision_depth', 100, 'get_accuracy', false, 're_thresh', -1, 'randomize_first', true, 'store_raw_pr', -1, 'verbose', 0);

precision_depth = min(precision_depth, length(labels));

if length(scores(:))==size(scores,2), scores = scores(:); end  % take care of row vectors

[nsamples, ns] = size(scores); 
assert(length(labels)==nsamples, 'label and score size does not agree!');
assert(nsamples>0, 'empty score??', 'warning');

% ns is the number of score-lists evaluated simultaneously (w.r.t the same label-seq.)
labels = labels(:);
ipos = find(any(labels*ones(1,length(pos_label)) == ones(nsamples,1)*pos_label(:)', 2)); 
ineg = find(any(labels*ones(1,length(neg_label)) == ones(nsamples,1)*neg_label(:)', 2));
assert(isempty(intersect(ipos, ineg)), 'pos and neg index sets should be mutually exclusive');
if length(ipos)+length(ineg)<length(labels) && verbose>0
    fprintf(1, ' %d samples of labels [%s] ignored\n', ...
        length(labels)-length(ipos)-length(ineg), num2str(setdiff(unique(labels), [pos_label neg_label])));
end
scores = scores([ipos; ineg], :);
labels = [true(length(ipos), 1); false(length(ineg), 1)];
nsamples = length(labels);

% assume binary classification with value(s) in pos_label as TRUE
if randomize_first
    rid = randperm(nsamples);
    labels = labels(rid);
    scores = scores(rid, :);
end
% rank the scores and shuffle the labels correspondingly
[~, si] = sort(scores);
%sscore = sscore(end:-1:1, :);
si = si(end:-1:1, :);
sorted_labels = zeros(nsamples, ns);
for i = 1:ns, 
    sorted_labels(:,i) = labels(si(:,i)); 
end
npos = sum(labels(:, 1));
prior = npos/nsamples;

pr = cumsum(sorted_labels, 1)./((1:nsamples)'*ones(1, ns));
ap = zeros(1, ns);
epr = 0;
re_index = []; % index where recall increments
if npos > 0
    epr = sum(sorted_labels(1:npos, :), 1)/npos ; % equal precision-recall
    re = cumsum(sorted_labels, 1) / npos;
    for i = 1 : ns
        ii = find(diff(re(:,i)));
        if re(1, i) > 0
            ire = [1; ii+1]; % find points where recall increases
        else
            ire = ii+1;
        end
        if ap_depth>0 && ap_depth<max(ire)
            ire = ire(ire<=ap_depth);
            ndenom = npos;
        elseif ~isempty(ap_range) && length(ap_range)==2
            ire = ire(ire/nsamples(i)>=ap_range(1) & ire/nsamples(i)<=ap_range(2));
            ndenom = length(ire);
        else
            ndenom = npos;
        end
        ap(i) = sum(pr(ire, i))/ndenom;
        re_index = unique([re_index; ire]);
    end
else
    re = zeros(size(scores));  % no TRUE example
end


if get_accuracy
    accuracy = zeros(nsamples, ns); % total # of correct decisions vs. cutoff point
    for j = 1 : nsamples
        accuracy(j, :) = sum(sorted_labels(1:j, :)) + sum(double(~sorted_labels(j+1 : end, :)));
    end
    accuracy = accuracy/nsamples;
    %accuracy(npos) = 1 - 2*prior*(1 - epr);
else
    accuracy = 'not computed';
end

fp = []; auc = []; f1 = [];
score_thresh = -inf*ones(1, ns);
if  store_raw_pr == 1
    ind = re_index;
elseif store_raw_pr > 0 
    if pr_nsample < nsamples && pr_nsample>0
        ind = round(nsamples*(.5:pr_nsample+1)/(pr_nsample+1));
    else
        ind = 1:nsamples;  % return full depth seq.
        fp =  ((1:nsamples)'*ones(1,ns)).*(1 - pr) / (nsamples-npos) ;
        auc = zeros(1, ns);
        f1 = zeros(1, ns);
        for i = 1 : ns
            ii = [1; find(diff(re(:,i))>0)+1; nsamples];
            auc(i) = trapz(fp(ii,i), re(ii,i));
            denom = (pr(:,i) + re(:,i)); di = find(denom>eps);
            if ~isempty(di)
                f1(i) = max( 2*pr(di,i).*re(di,i) ./ denom(di) );
            else
                f1(i) = 0;
            end
            
            % return score threshold for re > 
            if re_thresh>0
                ii = find(re(:,i) >= re_thresh);
                ind = si(ii(1), i);
                score_thresh(i) = scores(ind, i);
            end        
        end
        
    end
else    
    ind = [];
end
% assemble results
perf = struct('pr', pr(ind, :), 're', re(ind, :), 'fp', fp, 'auc', auc, 'f1', f1, 'ap', ap, 'ap_depth', ap_depth, 'ap_range', ap_range, ...
    'epr', epr, 'prior', prior, 'npos', npos, 'ns', nsamples, 'score_thresh', score_thresh, ...
    'p_at_d', pr(precision_depth, :), 'pr_dp', precision_depth, 'accuracy', accuracy);

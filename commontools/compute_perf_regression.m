function perf = compute_perf_regression(scores, ref, varargin)


perf.corr = corr(ref, scores); % pearson correlation

perf.tau = corr(ref, scores, 'type', 'kendall');

[n, ns] = size(scores);

perf.mse = - sum( (ref(:)*ones(1,ns) - scores ).^2, 1) / n; % minus, bigger is better
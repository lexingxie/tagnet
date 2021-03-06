
%% ----------- setup dir environment for tagnet experiments
addpath ../commontools/
addpath ../commontools/svm_wrap
addpath ../toolbox/lbfgsb3.0_mex1.1
addpath ../toolbox/libsvm-mat-3.0-1
%addpath ../toolbox/liblinear-1.92/matlab 
% input not sparse, don't use this
dbstop if error

[~,hostn] = system('hostname');
hostn = deblank(hostn);
% data dir pointing to "nuswide", with "db2" and "concept..." in the same parent dir
if strcmp(hostn, 'kinross') || strcmp(hostn, 'koves') || strcmp(hostn, 'infib2') || strcmp(hostn, 'infib1')
    data_dir = '/home/users/xlx/vault-xlx/imgnet-flickr/nuswide2';
elseif strcmp(hostn, 'cantabile')
    data_dir = '/home/xlx/data/imgnet-flickr/nuswide2';
elseif strcmp(hostn, 'xuming-UbuntuBox');
    data_dir = '/home/xlx/proj/imgnet-flickr/nuswide2';
else % mac os, clavier
    data_dir = '/Users/xlx/Documents/proj/imgnet-flickr/nuswide';
end

% set random state
tmp = version('-release');
v = str2double(tmp(1:4));
if v >= 2011
    rng(1);
else
    rand('twister', 1);
end

clear tmp v
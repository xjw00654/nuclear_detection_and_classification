close all; clear; clc;

addpath('/home/xjw/Matlab_lib/MCode/');

base_path = './data/CRC/Detection/';
list = dir([base_path 'img*']);

for i = 1:length(list)
    img = imread([base_path list(i).name '/' list(i).name '.bmp']);
    [~, H, E, ~, ~] = Deconvolve(img);
    imwrite(H, [base_path list(i).name '/' list(i).name '_H.png']);
    imwrite(E, [base_path list(i).name '/' list(i).name '_E.png']);
end

base_path = './data/CRC/test_data/';
list = dir([base_path 'img*']);

for i = 1:length(list)
    img = imread([base_path list(i).name]);
    [~, H, E, ~, ~] = Deconvolve(img);
    imwrite(H, [base_path 'H/' list(i).name(1:end - 4) '_H.png']);
    imwrite(E, [base_path 'E/' list(i).name(1:end - 4) '_E.png']);
end
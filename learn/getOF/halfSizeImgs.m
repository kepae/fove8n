clear all
close all
clc

sDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/learn/RAFT';

frames = dir([sDir filesep 'demo-frames' filesep '*.png']); frames = {frames.name};

for i = 1:length(frames)
    fn1 = [sDir filesep 'demo-frames' filesep frames{i}];
    img1 = imread(fn1);
    img2 = img1(1:2:end,1:2:end,:);
    fn2 = [sDir filesep 'halfSizeFrames' filesep frames{i}];
    imwrite(img2,fn2,'PNG');
end
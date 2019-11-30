%% this code reads occupancy grid from file
clear; close all; clc;

folder = '../data/';
filename = 'grid';

format = '%d ';
path = [folder,filename];
fileID = fopen(path,'r');
temp = fscanf(fileID,format);
fclose(fileID);

dim = round(length(temp)^(1/3));
grid = reshape(temp,[dim,dim,dim]);
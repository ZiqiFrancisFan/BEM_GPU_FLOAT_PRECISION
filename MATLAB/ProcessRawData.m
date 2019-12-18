function [grid,field] = ProcessRawData(grid_path,field_path,dark_len,dark_label)
%ProcessRawData Summary of this function goes here
%   Reads the raw data of an occupancy grid and a field to get matlab
%   matrices
fileID = fopen(grid_path,'r');
dim_sz = fread(fileID,3,'int');
grid = fread(fileID,dim_sz(1)*dim_sz(2)*dim_sz(3),'int');
fclose(fileID);

grid = reshape(grid,[dim_sz(1),dim_sz(2),dim_sz(3)]);

fileID = fopen(field_path,'r');
dim_sz = fread(fileID,3,'int');
field = fread(fileID,dim_sz(1)*dim_sz(2)*dim_sz(3),'single');
fclose(fileID);

field = reshape(field,[dim_sz(1),dim_sz(2),dim_sz(3)]);

len = (dim_sz(1)-dark_len)/2;

field(len+1:len+dark_len,len+1:len+dark_len) = nan;

%disp(min(min(field)));
disp([min(min(field)),max(max(field))]);

end


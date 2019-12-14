%% this code reads occupancy grid from file
clear; close all; clc;

folder = '../data/';
filename = 'vox';

path = [folder,filename];
fileID = fopen(path,'r');
dim_sz = fread(fileID,3,'int');
temp = fread(fileID,'int');
fclose(fileID);

grid = reshape(temp,[dim_sz(1),dim_sz(2),dim_sz(3)]);

for i = 1 : dim_sz(3)
    grid(:,:,i) = ConvertBoundary2SolidObject(grid(:,:,i));
end

folder = '../data/';
filename = 'loudness_3';

path = [folder,filename];
fileID = fopen(path,'r');
dim_sz = fread(fileID,3,'int');
temp = fread(fileID,'single');
fclose(fileID);

%for i=1:length(temp)/2
%    field(i) = complex(temp(2*i-1),temp(2*i));
%end


%field = abs(reshape(field,[dim,dim,dim]));
field = reshape(temp,[dim_sz(1),dim_sz(2),dim_sz(3)]);
field(find(grid==1)) = nan;



hFig1 = figure('units','normalized','outerposition',[0 0 1 1]);
hAx1 = axes(hFig1);
daspect(hAx1,[1,1,1]);

%hFig2 = figure('units','normalized','outerposition',[0 0 1 1]);
%hAx2 = axes(hFig2);
%daspect(hAx2,[1,1,1]);
%field(find(field>10)) = 5;
for i=1:dim_sz(3)
    %caxis([minVal maxVal]);
    imagesc(hAx1,field(:,:,i));
    daspect(hAx1,[1,1,1]);
    colorbar(hAx1);
    
    %imagesc(hAx2,grid(:,:,i));
    %daspect(hAx2,[1,1,1]);
    %pause(0.5);
end
xlabel(hAx1,'x');
ylabel(hAx1,'y');
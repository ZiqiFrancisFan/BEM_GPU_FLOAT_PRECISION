function [] = drawAARect(hAx,cnr,xLen,yLen)
%drawAARect Summary of this function goes here
%    explanation goes here
nods = cell(1,4);
nods{1} = cnr;
nods{2} = nods{1}+xLen*[1,0];
nods{3} = nods{2}+yLen*[0,1];
nods{4} = nods{3}-xLen*[1,0];
for i = 0 : 3
    x = [nods{i+1}(1), nods{mod(i+1,4)+1}(1)];
    y = [nods{i+1}(2), nods{mod(i+1,4)+1}(2)];
    hold(hAx,'on');
    plot(hAx,x,y,'LineWidth',2,'Color','k');
end

end

